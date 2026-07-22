"""
MLOps Pipeline Integration for Context Trimming
Provides production-ready interfaces and monitoring
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .trimmer import ContextTrimmer
from .budget_manager import BudgetManager
from .utils import TokenCounter, chunk_text, validate_context_chunks


@dataclass
class PipelineConfig:
    """Configuration for context trimming pipeline"""
    max_tokens: int = 4000
    selection_fraction: float = 0.7
    daily_budget: float = 100.0
    hourly_budget: float = 10.0
    cost_per_input_token: float = 0.0015 / 1000
    cost_per_output_token: float = 0.002 / 1000
    embedding_model: str = "all-MiniLM-L6-v2"
    tokenizer_model: str = "gpt-3.5-turbo"
    cache_size: int = 1000
    batch_size: int = 32
    max_workers: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 50
    enable_monitoring: bool = True


@dataclass
class PipelineResult:
    """Result from pipeline processing"""
    request_id: str
    selected_chunks: List[str]
    metadata: Dict[str, Any]
    budget_status: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class ContextPipeline:
    """
    Production-ready pipeline for context trimming with monitoring,
    budget management, and MLOps integration.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the context trimming pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize components
        self.trimmer = ContextTrimmer(
            max_tokens=config.max_tokens,
            selection_fraction=config.selection_fraction,
            embedding_model=config.embedding_model,
            tokenizer_model=config.tokenizer_model,
            cache_size=config.cache_size,
            batch_size=config.batch_size
        )

        self.budget_manager = BudgetManager(
            daily_budget=config.daily_budget,
            hourly_budget=config.hourly_budget,
            cost_per_input_token=config.cost_per_input_token,
            cost_per_output_token=config.cost_per_output_token
        )

        self.token_counter = TokenCounter(config.tokenizer_model)

        # Monitoring
        self.metrics = {
            "requests_processed": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
            "average_processing_time": 0.0,
            "success_rate": 1.0,
            "cache_hit_rate": 0.0
        }

        # Request tracking
        self.request_history: List[PipelineResult] = []

        logging.info("ContextPipeline initialized successfully")

    def process_request(
        self,
        context: Union[str, List[str]],
        query: str,
        request_id: Optional[str] = None,
        importance_weights: Optional[List[float]] = None,
        expected_output_tokens: int = 500
    ) -> PipelineResult:
        """
        Process a single context trimming request.

        Args:
            context: Context as string or list of chunks
            query: User query
            request_id: Optional request ID for tracking
            importance_weights: Optional importance weights for chunks
            expected_output_tokens: Expected output tokens for budget calculation

        Returns:
            PipelineResult with trimming results and metadata
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}"

        start_time = time.time()

        try:
            # Prepare context chunks
            if isinstance(context, str):
                context_chunks = chunk_text(
                    context,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap
                )
            else:
                context_chunks = validate_context_chunks(context)

            if not context_chunks:
                return PipelineResult(
                    request_id=request_id,
                    selected_chunks=[],
                    metadata={"error": "No valid context chunks"},
                    budget_status=self.budget_manager.get_budget_status(),
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="No valid context chunks provided"
                )

            # Estimate token usage
            estimated_input = sum(self.token_counter.count_tokens(chunk) for chunk in context_chunks)
            estimated_input += self.token_counter.count_tokens(query)

            # Check budget
            budget_ok, budget_message = self.budget_manager.check_budget_available(
                estimated_input, expected_output_tokens
            )

            if not budget_ok:
                return PipelineResult(
                    request_id=request_id,
                    selected_chunks=[],
                    metadata={"error": f"Budget constraint: {budget_message}"},
                    budget_status=self.budget_manager.get_budget_status(),
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"Budget constraint: {budget_message}"
                )

            # Perform trimming
            trimming_result = self.trimmer.trim_context(
                context_chunks=context_chunks,
                query=query,
                importance_weights=importance_weights
            )

            # Record actual usage
            actual_input_tokens = trimming_result["metadata"]["total_tokens"]
            usage_metrics = self.budget_manager.record_usage(
                actual_input_tokens, expected_output_tokens
            )

            # Calculate savings
            original_tokens = estimated_input + expected_output_tokens
            final_tokens = actual_input_tokens + expected_output_tokens
            tokens_saved = original_tokens - final_tokens
            cost_saved = tokens_saved * self.config.cost_per_input_token

            # Update metrics
            self._update_metrics(time.time() - start_time, tokens_saved, cost_saved, True)

            # Create result
            result = PipelineResult(
                request_id=request_id,
                selected_chunks=trimming_result["selected_chunks"],
                metadata={
                    **trimming_result["metadata"],
                    "original_chunk_count": len(context_chunks),
                    "tokens_saved": tokens_saved,
                    "cost_saved": cost_saved,
                    "budget_utilization": self.budget_manager.get_budget_status()["daily_utilization"]
                },
                budget_status=self.budget_manager.get_budget_status(),
                processing_time=time.time() - start_time,
                success=True
            )

            # Store for monitoring
            if self.config.enable_monitoring:
                self.request_history.append(result)
                # Keep only last 1000 requests
                if len(self.request_history) > 1000:
                    self.request_history = self.request_history[-1000:]

            return result

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logging.error(f"Request {request_id} failed: {error_msg}")

            self._update_metrics(time.time() - start_time, 0, 0.0, False)

            return PipelineResult(
                request_id=request_id,
                selected_chunks=[],
                metadata={"error": error_msg},
                budget_status=self.budget_manager.get_budget_status(),
                processing_time=time.time() - start_time,
                success=False,
                error_message=error_msg
            )

    def process_batch(
        self,
        requests: List[Dict[str, Any]],
        max_workers: Optional[int] = None
    ) -> List[PipelineResult]:
        """
        Process multiple requests in parallel.

        Args:
            requests: List of request dictionaries with 'context' and 'query' keys
            max_workers: Maximum number of worker threads

        Returns:
            List of PipelineResults
        """
        if max_workers is None:
            max_workers = self.config.max_workers

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_request = {}
            for i, request in enumerate(requests):
                future = executor.submit(
                    self.process_request,
                    request.get("context", ""),
                    request.get("query", ""),
                    request.get("request_id", f"batch_{i}"),
                    request.get("importance_weights"),
                    request.get("expected_output_tokens", 500)
                )
                future_to_request[future] = i

            # Collect results
            for future in as_completed(future_to_request):
                try:
                    result = future.result()
                    results.append((future_to_request[future], result))
                except Exception as e:
                    logging.error(f"Batch processing error: {e}")
                    results.append((future_to_request[future], None))

        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [result[1] for result in results if result[1] is not None]

    def get_health_status(self) -> Dict[str, Any]:
        """Get pipeline health status for monitoring."""
        budget_status = self.budget_manager.get_budget_status()
        cache_stats = self.trimmer.get_stats()

        return {
            "status": "healthy" if budget_status["status"] != "BUDGET_EXCEEDED" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "budget": budget_status,
            "cache": cache_stats,
            "metrics": self.metrics,
            "config": asdict(self.config)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed pipeline metrics."""
        recent_results = self.request_history[-100:] if self.request_history else []

        if recent_results:
            avg_processing_time = sum(r.processing_time for r in recent_results) / len(recent_results)
            success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)

            # Token savings statistics
            token_savings = [r.metadata.get("tokens_saved", 0) for r in recent_results if r.success]
            avg_tokens_saved = sum(token_savings) / len(token_savings) if token_savings else 0

            # Cost savings statistics
            cost_savings = [r.metadata.get("cost_saved", 0.0) for r in recent_results if r.success]
            avg_cost_saved = sum(cost_savings) / len(cost_savings) if cost_savings else 0.0
        else:
            avg_processing_time = 0.0
            success_rate = 1.0
            avg_tokens_saved = 0
            avg_cost_saved = 0.0

        return {
            "requests_processed": self.metrics["requests_processed"],
            "avg_processing_time": avg_processing_time,
            "success_rate": success_rate,
            "avg_tokens_saved": avg_tokens_saved,
            "avg_cost_saved": avg_cost_saved,
            "total_cost_saved": self.metrics["total_cost_saved"],
            "cache_hit_rate": self.trimmer.embedding_cache.get_stats()["hit_rate"],
            "recent_requests": len(recent_results),
            "budget_utilization": self.budget_manager.get_budget_status()["daily_utilization"]
        }

    def configure_alerts(self, alert_callback: Callable[[Dict[str, Any]], None]):
        """
        Configure alert callback for budget and performance issues.

        Args:
            alert_callback: Function to call when alerts are triggered
        """
        self.alert_callback = alert_callback

    def _update_metrics(self, processing_time: float, tokens_saved: int, cost_saved: float, success: bool):
        """Update internal metrics."""
        self.metrics["requests_processed"] += 1

        if success:
            self.metrics["total_tokens_saved"] += tokens_saved
            self.metrics["total_cost_saved"] += cost_saved

        # Update running averages
        total_requests = self.metrics["requests_processed"]
        self.metrics["average_processing_time"] = (
            (self.metrics["average_processing_time"] * (total_requests - 1) + processing_time) / total_requests
        )

        successful_requests = sum(1 for r in self.request_history if r.success) if self.request_history else (1 if success else 0)
        self.metrics["success_rate"] = successful_requests / total_requests

    def export_config(self, file_path: str):
        """Export current configuration to file."""
        config_dict = asdict(self.config)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_config_file(cls, file_path: str) -> 'ContextPipeline':
        """Load pipeline from configuration file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)

        config = PipelineConfig(**config_dict)
        return cls(config)