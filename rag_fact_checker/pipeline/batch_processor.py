import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 10  # Number of items to process in single API call
    max_workers: int = 3  # Number of concurrent workers
    max_retries: int = 3  # Maximum retries for failed items
    retry_delay: float = 1.0  # Delay between retries in seconds
    timeout: float | None = None  # Timeout per batch in seconds


@dataclass
class BatchResult(Generic[T]):
    """Result of batch processing operation."""

    results: list[T]
    failed_indices: list[int]
    errors: list[Exception]
    total_time: float
    api_calls_made: int


class BatchProcessor(Generic[T, U]):
    """
    Generic batch processor for handling multiple items efficiently.

    Provides both synchronous and asynchronous batch processing with:
    - Configurable batch sizes
    - Concurrent processing
    - Automatic retries
    - Error handling and reporting
    """

    def __init__(self, config: BatchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api_calls_made = 0

    def process_batch_sync(
        self,
        items: list[T],
        processor_func: Callable[[list[T]], list[U]],
        item_name: str = "items",
    ) -> BatchResult[U]:
        """
        Process items in batches synchronously with concurrent workers.

        Args:
            items: List of items to process
            processor_func: Function that processes a batch of items
            item_name: Name for logging purposes

        Returns:
            BatchResult containing all results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting batch processing of {len(items)} {item_name}")

        # Split items into batches
        batches = self._create_batches(items)

        results = []
        failed_indices = []
        errors = []

        # Process batches concurrently
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(
                    self._process_single_batch_with_retry,
                    batch,
                    batch_idx,
                    processor_func,
                ): (batch, batch_idx)
                for batch_idx, batch in enumerate(batches)
            }

            # Collect results
            for future in as_completed(future_to_batch):
                batch, batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=self.config.timeout)
                    results.extend(batch_results)
                    self.api_calls_made += 1
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx} failed completely: {str(e)}")
                    # Add failed indices for this entire batch
                    start_idx = batch_idx * self.config.batch_size
                    batch_failed_indices = list(
                        range(start_idx, start_idx + len(batch))
                    )
                    failed_indices.extend(batch_failed_indices)
                    errors.append(e)

        total_time = time.time() - start_time

        self.logger.info(
            f"Batch processing completed: {len(results)} successful, "
            f"{len(failed_indices)} failed, {self.api_calls_made} API calls, "
            f"{total_time:.2f}s total"
        )

        return BatchResult(
            results=results,
            failed_indices=failed_indices,
            errors=errors,
            total_time=total_time,
            api_calls_made=self.api_calls_made,
        )

    async def process_batch_async(
        self,
        items: list[T],
        async_processor_func: Callable[[list[T]], list[U]],
        item_name: str = "items",
    ) -> BatchResult[U]:
        """
        Process items in batches asynchronously.

        Args:
            items: List of items to process
            async_processor_func: Async function that processes a batch of items
            item_name: Name for logging purposes

        Returns:
            BatchResult containing all results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting async batch processing of {len(items)} {item_name}")

        # Split items into batches
        batches = self._create_batches(items)

        # Create semaphore to limit concurrent batches
        semaphore = asyncio.Semaphore(self.config.max_workers)

        # Process all batches concurrently
        tasks = [
            self._process_single_batch_async_with_retry(
                batch, batch_idx, async_processor_func, semaphore
            )
            for batch_idx, batch in enumerate(batches)
        ]

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful results from errors
        results = []
        failed_indices = []
        errors = []

        for batch_idx, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                self.logger.error(
                    f"Async batch {batch_idx} failed: {str(batch_result)}"
                )
                # Add failed indices for this entire batch
                start_idx = batch_idx * self.config.batch_size
                batch_size = len(batches[batch_idx])
                batch_failed_indices = list(range(start_idx, start_idx + batch_size))
                failed_indices.extend(batch_failed_indices)
                errors.append(batch_result)
            else:
                results.extend(batch_result)
                self.api_calls_made += 1

        total_time = time.time() - start_time

        self.logger.info(
            f"Async batch processing completed: {len(results)} successful, "
            f"{len(failed_indices)} failed, {self.api_calls_made} API calls, "
            f"{total_time:.2f}s total"
        )

        return BatchResult(
            results=results,
            failed_indices=failed_indices,
            errors=errors,
            total_time=total_time,
            api_calls_made=self.api_calls_made,
        )

    def _create_batches(self, items: list[T]) -> list[list[T]]:
        """Split items into batches of configured size."""
        batches = []
        for i in range(0, len(items), self.config.batch_size):
            batch = items[i : i + self.config.batch_size]
            batches.append(batch)
        return batches

    def _process_single_batch_with_retry(
        self,
        batch: list[T],
        batch_idx: int,
        processor_func: Callable[[list[T]], list[U]],
    ) -> list[U]:
        """Process a single batch with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(
                    f"Processing batch {batch_idx}, attempt {attempt + 1}"
                )
                return processor_func(batch)
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Batch {batch_idx} attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(
                        self.config.retry_delay * (attempt + 1)
                    )  # Exponential backoff

        # All retries failed
        self.logger.error(
            f"Batch {batch_idx} failed after {self.config.max_retries} attempts"
        )
        raise last_exception

    async def _process_single_batch_async_with_retry(
        self,
        batch: list[T],
        batch_idx: int,
        async_processor_func: Callable[[list[T]], list[U]],
        semaphore: asyncio.Semaphore,
    ) -> list[U]:
        """Process a single batch asynchronously with retry logic."""
        async with semaphore:
            last_exception = None

            for attempt in range(self.config.max_retries):
                try:
                    self.logger.debug(
                        f"Processing async batch {batch_idx}, attempt {attempt + 1}"
                    )
                    return await async_processor_func(batch)
                except Exception as e:
                    last_exception = e
                    self.logger.warning(
                        f"Async batch {batch_idx} attempt {attempt + 1} failed: {str(e)}"
                    )
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))

            # All retries failed
            self.logger.error(
                f"Async batch {batch_idx} failed after {self.config.max_retries} attempts"
            )
            raise last_exception


class BatchProcessingMixin:
    """
    Mixin class to add batch processing capabilities to existing components.

    Can be mixed into any class that has:
    - self.logger: logging.Logger
    - self.config: Config (with batch_config attribute)
    """

    def get_batch_processor(self) -> BatchProcessor:
        """Get configured batch processor instance."""
        batch_config = getattr(self.config, "batch_config", BatchConfig())
        return BatchProcessor(batch_config, self.logger)

    def process_items_in_batches_sync(
        self,
        items: list[T],
        processor_func: Callable[[list[T]], list[U]],
        item_name: str = "items",
    ) -> BatchResult[U]:
        """Convenience method for synchronous batch processing."""
        batch_processor = self.get_batch_processor()
        return batch_processor.process_batch_sync(items, processor_func, item_name)

    async def process_items_in_batches_async(
        self,
        items: list[T],
        async_processor_func: Callable[[list[T]], list[U]],
        item_name: str = "items",
    ) -> BatchResult[U]:
        """Convenience method for asynchronous batch processing."""
        batch_processor = self.get_batch_processor()
        return await batch_processor.process_batch_async(
            items, async_processor_func, item_name
        )
