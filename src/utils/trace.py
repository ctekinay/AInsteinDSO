"""
Comprehensive tracing utility for EA Assistant pipeline.

This module provides detailed tracing and logging to understand exactly which
modules are called and in what order during query processing.
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Single trace event in the pipeline."""
    trace_id: str
    event_id: str
    timestamp: str
    module: str
    function: str
    event_type: str  # START, END, INFO, ERROR
    duration_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineTracer:
    """
    Comprehensive tracer for EA Assistant pipeline.

    Tracks every function call, timing, and data flow through the system.
    """

    def __init__(self):
        self.traces: Dict[str, List[TraceEvent]] = {}
        self.active_traces: Dict[str, Dict[str, float]] = {}

    def start_trace(self, trace_id: Optional[str] = None) -> str:
        """Start a new trace for a query."""
        if not trace_id:
            trace_id = str(uuid4())[:8]

        self.traces[trace_id] = []
        self.active_traces[trace_id] = {}

        self._add_event(
            trace_id=trace_id,
            module="tracer",
            function="start_trace",
            event_type="START",
            details={"trace_started": trace_id}
        )

        logger.info(f"🔍 TRACE STARTED: {trace_id}")
        return trace_id

    def _add_event(
        self,
        trace_id: str,
        module: str,
        function: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None
    ) -> None:
        """Add an event to the trace."""
        event = TraceEvent(
            trace_id=trace_id,
            event_id=str(uuid4())[:8],
            timestamp=datetime.utcnow().isoformat(),
            module=module,
            function=function,
            event_type=event_type,
            duration_ms=duration_ms,
            details=details,
            error=error
        )

        if trace_id not in self.traces:
            self.traces[trace_id] = []

        self.traces[trace_id].append(event)

        # Log the event
        if event_type == "START":
            logger.info(f"🚀 [{trace_id}] {module}.{function} STARTED")
        elif event_type == "END":
            duration_str = f" ({duration_ms:.1f}ms)" if duration_ms else ""
            logger.info(f"✅ [{trace_id}] {module}.{function} COMPLETED{duration_str}")
        elif event_type == "INFO":
            logger.info(f"ℹ️  [{trace_id}] {module}.{function}: {details}")
        elif event_type == "ERROR":
            logger.error(f"❌ [{trace_id}] {module}.{function} ERROR: {error}")

    @contextmanager
    def trace_function(self, trace_id: str, module: str, function: str, **details):
        """Context manager to trace a function call."""
        start_key = f"{module}.{function}"
        start_time = time.perf_counter()

        self._add_event(
            trace_id=trace_id,
            module=module,
            function=function,
            event_type="START",
            details=details
        )

        self.active_traces[trace_id][start_key] = start_time

        try:
            yield
        except Exception as e:
            self._add_event(
                trace_id=trace_id,
                module=module,
                function=function,
                event_type="ERROR",
                error=str(e)
            )
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            self._add_event(
                trace_id=trace_id,
                module=module,
                function=function,
                event_type="END",
                duration_ms=duration_ms
            )

            if start_key in self.active_traces[trace_id]:
                del self.active_traces[trace_id][start_key]

    def trace_info(self, trace_id: str, module: str, function: str, **details):
        """Log an info event in the trace."""
        self._add_event(
            trace_id=trace_id,
            module=module,
            function=function,
            event_type="INFO",
            details=details
        )

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get a summary of the trace."""
        if trace_id not in self.traces:
            return {"error": f"Trace {trace_id} not found"}

        events = self.traces[trace_id]
        if not events:
            return {"error": f"No events in trace {trace_id}"}

        start_time = datetime.fromisoformat(events[0].timestamp)
        end_time = datetime.fromisoformat(events[-1].timestamp)
        total_duration = (end_time - start_time).total_seconds() * 1000

        # Group events by module
        module_calls = {}
        for event in events:
            if event.module not in module_calls:
                module_calls[event.module] = []
            module_calls[event.module].append({
                "function": event.function,
                "event_type": event.event_type,
                "duration_ms": event.duration_ms,
                "details": event.details
            })

        # Calculate step durations
        step_durations = {}
        for event in events:
            if event.event_type == "END" and event.duration_ms:
                step_key = f"{event.module}.{event.function}"
                step_durations[step_key] = event.duration_ms

        return {
            "trace_id": trace_id,
            "total_duration_ms": total_duration,
            "total_events": len(events),
            "modules_called": list(module_calls.keys()),
            "module_calls": module_calls,
            "step_durations": step_durations,
            "timeline": [
                {
                    "timestamp": event.timestamp,
                    "module": event.module,
                    "function": event.function,
                    "event_type": event.event_type,
                    "duration_ms": event.duration_ms
                }
                for event in events
            ]
        }

    def print_trace_report(self, trace_id: str) -> None:
        """Print a formatted trace report to console."""
        summary = self.get_trace_summary(trace_id)

        if "error" in summary:
            print(f"❌ {summary['error']}")
            return

        print(f"\n🔍 TRACE REPORT: {trace_id}")
        print("=" * 60)
        print(f"📊 Total Duration: {summary['total_duration_ms']:.1f}ms")
        print(f"📈 Total Events: {summary['total_events']}")
        print(f"🔧 Modules Called: {', '.join(summary['modules_called'])}")

        print(f"\n⏱️  STEP DURATIONS:")
        for step, duration in sorted(summary['step_durations'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"   {step}: {duration:.1f}ms")

        print(f"\n📝 TIMELINE:")
        for event in summary['timeline']:
            icon = {
                'START': '🚀',
                'END': '✅',
                'INFO': 'ℹ️ ',
                'ERROR': '❌'
            }.get(event['event_type'], '⚡')

            duration_str = f" ({event['duration_ms']:.1f}ms)" if event['duration_ms'] else ""
            print(f"   {icon} {event['module']}.{event['function']}{duration_str}")

        print("=" * 60)

    def save_trace_to_file(self, trace_id: str, filepath: str) -> None:
        """Save trace to JSON file."""
        summary = self.get_trace_summary(trace_id)

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Trace saved to: {filepath}")


# Global tracer instance
_tracer = PipelineTracer()


def get_tracer() -> PipelineTracer:
    """Get the global tracer instance."""
    return _tracer


def trace_function(module: str, function: str, **details):
    """Decorator to automatically trace function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to get trace_id from args or kwargs
            trace_id = None

            # Check if first arg has trace_id attribute (for class methods)
            if args and hasattr(args[0], '_trace_id'):
                trace_id = args[0]._trace_id

            # Check kwargs
            if 'trace_id' in kwargs:
                trace_id = kwargs['trace_id']

            if not trace_id:
                # Start new trace if none found
                trace_id = _tracer.start_trace()

            with _tracer.trace_function(trace_id, module, function, **details):
                return func(*args, **kwargs)

        return wrapper
    return decorator