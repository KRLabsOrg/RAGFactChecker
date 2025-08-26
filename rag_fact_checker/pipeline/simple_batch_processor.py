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
class SimpleBatchConfig:
    """Configuration for simple batch processing."""

    max_workers: int = 5  # Number of concurrent threads
    max_retries: int = 3  # Maximum retries for failed items
    retry_delay: float = 1.0  # Delay between retries in seconds
    timeout: float | None = None  # Timeout per individual call


@dataclass
class SimpleBatchResult(Generic[T]):
    """Result of simple batch processing operation."""

    results: list[T]
    failed_indices: list[int]
    errors: list[Exception]
    total_time: float
    successful_count: int
    failed_count: int


class SimpleBatchProcessor:
    """
    Simple batch processor that makes multiple concurrent calls to existing single-item methods.

    This doesn't change prompts or pack multiple items into single API calls.
    Instead, it runs existing methods concurrently using threading.
    """

    def __init__(self, config: SimpleBatchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def process_batch(
        self, items: list[T], processor_func: Callable[[T], U], item_name: str = "items"
    ) -> SimpleBatchResult[U]:
        """
        Process items concurrently using existing single-item processor function.

        Args:
            items: List of items to process
            processor_func: Function that processes a single item
            item_name: Name for logging purposes

        Returns:
            SimpleBatchResult containing all results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting batch processing: {len(items)} {item_name}")
        self.logger.info(
            f"Batch config: max_workers={self.config.max_workers}, max_retries={self.config.max_retries}, retry_delay={self.config.retry_delay}s"
        )

        results = [None] * len(items)
        failed_indices = []
        errors = []
        completed_count = 0

        # Process items concurrently
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all items
            future_to_index = {
                executor.submit(
                    self._process_single_item_with_retry, item, idx, processor_func
                ): idx
                for idx, item in enumerate(items)
            }

            # Collect results
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result(timeout=self.config.timeout)
                    results[idx] = result
                    completed_count += 1
                    if (
                        completed_count % max(1, len(items) // 10) == 0
                    ):  # Log progress every 10%
                        progress_pct = (completed_count * 100) // len(items)
                        self.logger.info(
                            f"Progress: {completed_count}/{len(items)} items completed ({progress_pct}%)"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Item {idx + 1}/{len(items)} failed completely: {str(e)}"
                    )
                    failed_indices.append(idx)
                    errors.append(e)
                    completed_count += 1

        # Filter out None results
        successful_results = [r for r in results if r is not None]

        total_time = time.time() - start_time

        self.logger.info(
            f"Batch processing completed in {total_time:.2f}s: "
            f"{len(successful_results)} successful, {len(failed_indices)} failed"
        )
        if len(failed_indices) > 0:
            self.logger.error(f"Failed item indices: {failed_indices}")

        return SimpleBatchResult(
            results=successful_results,
            failed_indices=failed_indices,
            errors=errors,
            total_time=total_time,
            successful_count=len(successful_results),
            failed_count=len(failed_indices),
        )

    async def process_batch_async(
        self,
        items: list[T],
        async_processor_func: Callable[[T], U],
        item_name: str = "items",
    ) -> SimpleBatchResult[U]:
        """
        Process items concurrently using async processor function.

        Args:
            items: List of items to process
            async_processor_func: Async function that processes a single item
            item_name: Name for logging purposes

        Returns:
            SimpleBatchResult containing all results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting async batch processing: {len(items)} {item_name}")
        self.logger.info(
            f"Async batch config: max_workers={self.config.max_workers}, max_retries={self.config.max_retries}, retry_delay={self.config.retry_delay}s"
        )

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.config.max_workers)

        # Process all items concurrently
        tasks = [
            self._process_single_item_async_with_retry(
                item, idx, async_processor_func, semaphore
            )
            for idx, item in enumerate(items)
        ]

        # Wait for all items to complete
        item_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful results from errors
        results = []
        failed_indices = []
        errors = []

        for idx, item_result in enumerate(item_results):
            if isinstance(item_result, Exception):
                self.logger.error(f"Async item {idx} failed: {str(item_result)}")
                failed_indices.append(idx)
                errors.append(item_result)
            else:
                results.append(item_result)

        total_time = time.time() - start_time

        self.logger.info(
            f"Async batch processing completed in {total_time:.2f}s: "
            f"{len(results)} successful, {len(failed_indices)} failed"
        )
        if len(failed_indices) > 0:
            self.logger.error(f"Failed item indices: {failed_indices}")

        return SimpleBatchResult(
            results=results,
            failed_indices=failed_indices,
            errors=errors,
            total_time=total_time,
            successful_count=len(results),
            failed_count=len(failed_indices),
        )

    def _process_single_item_with_retry(
        self, item: T, item_idx: int, processor_func: Callable[[T], U]
    ) -> U:
        """Process a single item with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(
                    f"Processing item {item_idx + 1}, attempt {attempt + 1}/{self.config.max_retries}"
                )
                return processor_func(item)
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Item {item_idx + 1} failed attempt {attempt + 1}/{self.config.max_retries}: {str(e)}"
                )
                if attempt < self.config.max_retries - 1:
                    retry_delay = self.config.retry_delay * (
                        attempt + 1
                    )  # Exponential backoff
                    self.logger.debug(
                        f"Item {item_idx + 1}: Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)

        # All retries failed
        self.logger.error(
            f"Item {item_idx + 1} failed after {self.config.max_retries} attempts"
        )
        raise last_exception

    async def _process_single_item_async_with_retry(
        self,
        item: T,
        item_idx: int,
        async_processor_func: Callable[[T], U],
        semaphore: asyncio.Semaphore,
    ) -> U:
        """Process a single item asynchronously with retry logic."""
        async with semaphore:
            last_exception = None

            for attempt in range(self.config.max_retries):
                try:
                    self.logger.debug(
                        f"Processing async item {item_idx + 1}, attempt {attempt + 1}/{self.config.max_retries}"
                    )
                    return await async_processor_func(item)
                except Exception as e:
                    last_exception = e
                    self.logger.warning(
                        f"Async item {item_idx + 1} failed attempt {attempt + 1}/{self.config.max_retries}: {str(e)}"
                    )
                    if attempt < self.config.max_retries - 1:
                        retry_delay = self.config.retry_delay * (attempt + 1)
                        self.logger.debug(
                            f"Async item {item_idx + 1}: Retrying in {retry_delay}s..."
                        )
                        await asyncio.sleep(retry_delay)

            # All retries failed
            self.logger.error(
                f"Async item {item_idx + 1} failed after {self.config.max_retries} attempts"
            )
            raise last_exception


class SimpleBatchProcessingMixin:
    """
    Mixin class to add simple batch processing capabilities to existing components.

    Can be mixed into any class that has:
    - self.logger: logging.Logger
    - self.config: Config (with simple_batch_config attribute)
    """

    def get_simple_batch_processor(self) -> SimpleBatchProcessor:
        """Get configured simple batch processor instance."""
        batch_config = getattr(self.config, "simple_batch_config", SimpleBatchConfig())
        return SimpleBatchProcessor(batch_config, self.logger)

    def process_items_concurrently(
        self, items: list[T], processor_func: Callable[[T], U], item_name: str = "items"
    ) -> SimpleBatchResult[U]:
        """Convenience method for concurrent processing."""
        batch_processor = self.get_simple_batch_processor()
        return batch_processor.process_batch(items, processor_func, item_name)

    async def process_items_concurrently_async(
        self,
        items: list[T],
        async_processor_func: Callable[[T], U],
        item_name: str = "items",
    ) -> SimpleBatchResult[U]:
        """Convenience method for async concurrent processing."""
        batch_processor = self.get_simple_batch_processor()
        return await batch_processor.process_batch_async(
            items, async_processor_func, item_name
        )
