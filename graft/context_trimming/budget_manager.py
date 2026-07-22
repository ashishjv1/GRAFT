"""
Budget Management for Context Trimming
Handles cost tracking and budget constraints for LLM inference
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class CostMetrics:
    """Cost tracking metrics"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    requests: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BudgetManager:
    """
    Manages token budgets and cost tracking for context trimming operations.
    Supports multiple pricing models and budget enforcement.
    """

    def __init__(
        self,
        daily_budget: float = 100.0,
        hourly_budget: float = 10.0,
        cost_per_input_token: float = 0.0015 / 1000,  # GPT-3.5 pricing
        cost_per_output_token: float = 0.002 / 1000,
        alert_threshold: float = 0.8,  # Alert at 80% budget usage
        max_tokens_per_request: int = 4096
    ):
        """
        Initialize Budget Manager.

        Args:
            daily_budget: Maximum daily spending limit
            hourly_budget: Maximum hourly spending limit
            cost_per_input_token: Cost per input token
            cost_per_output_token: Cost per output token
            alert_threshold: Threshold for budget alerts (0.0 to 1.0)
            max_tokens_per_request: Maximum tokens per individual request
        """
        self.daily_budget = daily_budget
        self.hourly_budget = hourly_budget
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.alert_threshold = alert_threshold
        self.max_tokens_per_request = max_tokens_per_request

        # Usage tracking
        self.usage_history: List[CostMetrics] = []
        self.daily_usage = CostMetrics()
        self.hourly_usage = CostMetrics()

        # Alert state
        self.alerts_sent = set()

        logging.info(f"BudgetManager initialized - Daily: ${daily_budget:.2f}, Hourly: ${hourly_budget:.2f}")

    def check_budget_available(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int = 0
    ) -> Tuple[bool, str]:
        """
        Check if budget is available for the estimated token usage.

        Args:
            estimated_input_tokens: Expected input tokens
            estimated_output_tokens: Expected output tokens

        Returns:
            (is_allowed, reason)
        """
        estimated_cost = self._calculate_cost(estimated_input_tokens, estimated_output_tokens)

        # Check token limits
        total_tokens = estimated_input_tokens + estimated_output_tokens
        if total_tokens > self.max_tokens_per_request:
            return False, f"Request exceeds max tokens ({total_tokens} > {self.max_tokens_per_request})"

        # Update current usage counters
        self._update_usage_counters()

        # Check daily budget
        if self.daily_usage.total_cost + estimated_cost > self.daily_budget:
            remaining = self.daily_budget - self.daily_usage.total_cost
            return False, f"Would exceed daily budget. Remaining: ${remaining:.4f}, Needed: ${estimated_cost:.4f}"

        # Check hourly budget
        if self.hourly_usage.total_cost + estimated_cost > self.hourly_budget:
            remaining = self.hourly_budget - self.hourly_usage.total_cost
            return False, f"Would exceed hourly budget. Remaining: ${remaining:.4f}, Needed: ${estimated_cost:.4f}"

        return True, "Budget available"

    def record_usage(
        self,
        actual_input_tokens: int,
        actual_output_tokens: int,
        response_time: float = 0.0
    ) -> CostMetrics:
        """
        Record actual token usage after API call.

        Args:
            actual_input_tokens: Actual input tokens used
            actual_output_tokens: Actual output tokens received
            response_time: API response time in seconds

        Returns:
            CostMetrics for this usage
        """
        cost = self._calculate_cost(actual_input_tokens, actual_output_tokens)

        metrics = CostMetrics(
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            total_cost=cost,
            requests=1,
            timestamp=datetime.now()
        )

        # Update tracking
        self.usage_history.append(metrics)
        self._update_usage_counters()

        # Check for alerts
        self._check_budget_alerts()

        logging.debug(f"Usage recorded: ${cost:.4f} ({actual_input_tokens} in, {actual_output_tokens} out)")

        return metrics

    def get_budget_status(self) -> Dict:
        """Get current budget status and usage."""
        self._update_usage_counters()

        daily_remaining = max(0, self.daily_budget - self.daily_usage.total_cost)
        hourly_remaining = max(0, self.hourly_budget - self.hourly_usage.total_cost)

        daily_utilization = self.daily_usage.total_cost / self.daily_budget
        hourly_utilization = self.hourly_usage.total_cost / self.hourly_budget

        return {
            "daily_budget": self.daily_budget,
            "daily_used": self.daily_usage.total_cost,
            "daily_remaining": daily_remaining,
            "daily_utilization": daily_utilization,
            "hourly_budget": self.hourly_budget,
            "hourly_used": self.hourly_usage.total_cost,
            "hourly_remaining": hourly_remaining,
            "hourly_utilization": hourly_utilization,
            "total_requests_today": self.daily_usage.requests,
            "total_requests_hour": self.hourly_usage.requests,
            "status": self._get_status_message(daily_utilization, hourly_utilization)
        }

    def optimize_for_budget(
        self,
        estimated_input_tokens: int,
        target_output_tokens: int = 500,
        priority: str = "cost"  # "cost", "quality", "balanced"
    ) -> Dict[str, int]:
        """
        Optimize token allocation for budget constraints.

        Args:
            estimated_input_tokens: Current estimated input tokens
            target_output_tokens: Desired output tokens
            priority: Optimization priority

        Returns:
            Optimized token allocation
        """
        self._update_usage_counters()

        # Calculate remaining budgets
        daily_remaining = self.daily_budget - self.daily_usage.total_cost
        hourly_remaining = self.hourly_budget - self.hourly_usage.total_cost
        effective_budget = min(daily_remaining, hourly_remaining)

        if effective_budget <= 0:
            return {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0}

        # Optimization strategies
        if priority == "cost":
            # Minimize cost, potentially reducing quality
            max_total_cost = effective_budget * 0.9  # Leave 10% buffer
            optimized = self._optimize_for_cost(estimated_input_tokens, target_output_tokens, max_total_cost)

        elif priority == "quality":
            # Maximize quality within budget
            optimized = self._optimize_for_quality(estimated_input_tokens, target_output_tokens, effective_budget)

        else:  # balanced
            # Balance cost and quality
            target_cost = effective_budget * 0.7  # Use 70% of remaining budget
            optimized = self._optimize_balanced(estimated_input_tokens, target_output_tokens, target_cost)

        return optimized

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for token usage."""
        input_cost = input_tokens * self.cost_per_input_token
        output_cost = output_tokens * self.cost_per_output_token
        return input_cost + output_cost

    def _update_usage_counters(self):
        """Update daily and hourly usage counters."""
        now = datetime.now()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        hour_start = now.replace(minute=0, second=0, microsecond=0)

        # Reset counters
        self.daily_usage = CostMetrics()
        self.hourly_usage = CostMetrics()

        # Sum usage for current periods
        for usage in self.usage_history:
            if usage.timestamp >= day_start:
                self.daily_usage.input_tokens += usage.input_tokens
                self.daily_usage.output_tokens += usage.output_tokens
                self.daily_usage.total_cost += usage.total_cost
                self.daily_usage.requests += usage.requests

            if usage.timestamp >= hour_start:
                self.hourly_usage.input_tokens += usage.input_tokens
                self.hourly_usage.output_tokens += usage.output_tokens
                self.hourly_usage.total_cost += usage.total_cost
                self.hourly_usage.requests += usage.requests

    def _check_budget_alerts(self):
        """Check if budget alerts should be sent."""
        status = self.get_budget_status()

        # Daily budget alert
        if (status["daily_utilization"] >= self.alert_threshold and
            "daily_alert" not in self.alerts_sent):
            logging.warning(f"Daily budget alert: {status['daily_utilization']:.1%} used")
            self.alerts_sent.add("daily_alert")

        # Hourly budget alert
        if (status["hourly_utilization"] >= self.alert_threshold and
            "hourly_alert" not in self.alerts_sent):
            logging.warning(f"Hourly budget alert: {status['hourly_utilization']:.1%} used")
            self.alerts_sent.add("hourly_alert")

        # Reset alerts at new time periods
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:  # New day
            self.alerts_sent.discard("daily_alert")
        if now.minute == 0:  # New hour
            self.alerts_sent.discard("hourly_alert")

    def _get_status_message(self, daily_util: float, hourly_util: float) -> str:
        """Get status message based on utilization."""
        max_util = max(daily_util, hourly_util)

        if max_util >= 1.0:
            return "BUDGET_EXCEEDED"
        elif max_util >= 0.9:
            return "BUDGET_CRITICAL"
        elif max_util >= 0.7:
            return "BUDGET_HIGH"
        elif max_util >= 0.5:
            return "BUDGET_MEDIUM"
        else:
            return "BUDGET_OK"

    def _optimize_for_cost(self, input_tokens: int, output_tokens: int, max_cost: float) -> Dict[str, int]:
        """Optimize for minimum cost."""
        # Reduce tokens if necessary to fit budget
        current_cost = self._calculate_cost(input_tokens, output_tokens)

        if current_cost <= max_cost:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": current_cost
            }

        # Reduce input tokens first (usually cheaper to trim context)
        reduction_factor = max_cost / current_cost
        optimized_input = int(input_tokens * reduction_factor * 0.8)  # Aggressive reduction
        optimized_output = int(output_tokens * reduction_factor * 1.2)  # Preserve output more

        # Ensure we stay within budget
        while self._calculate_cost(optimized_input, optimized_output) > max_cost:
            optimized_input = max(1, int(optimized_input * 0.9))
            optimized_output = max(1, int(optimized_output * 0.95))

        return {
            "input_tokens": optimized_input,
            "output_tokens": optimized_output,
            "estimated_cost": self._calculate_cost(optimized_input, optimized_output)
        }

    def _optimize_for_quality(self, input_tokens: int, output_tokens: int, max_budget: float) -> Dict[str, int]:
        """Optimize for maximum quality within budget."""
        max_cost = max_budget * 0.95  # Leave small buffer

        # Try to preserve output tokens (quality) while reducing input
        current_cost = self._calculate_cost(input_tokens, output_tokens)

        if current_cost <= max_cost:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": current_cost
            }

        # Reduce input tokens while keeping output tokens
        max_input_cost = max_cost - (output_tokens * self.cost_per_output_token)
        optimized_input = int(max_input_cost / self.cost_per_input_token)

        return {
            "input_tokens": max(1, optimized_input),
            "output_tokens": output_tokens,
            "estimated_cost": self._calculate_cost(max(1, optimized_input), output_tokens)
        }

    def _optimize_balanced(self, input_tokens: int, output_tokens: int, target_cost: float) -> Dict[str, int]:
        """Balance cost and quality optimization."""
        current_cost = self._calculate_cost(input_tokens, output_tokens)

        if current_cost <= target_cost:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": current_cost
            }

        # Proportional reduction
        reduction_factor = target_cost / current_cost
        optimized_input = int(input_tokens * reduction_factor)
        optimized_output = int(output_tokens * reduction_factor)

        return {
            "input_tokens": max(1, optimized_input),
            "output_tokens": max(1, optimized_output),
            "estimated_cost": self._calculate_cost(max(1, optimized_input), max(1, optimized_output))
        }