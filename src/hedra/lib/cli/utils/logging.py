"""
Enhanced logging utilities for CLI operations with per-job logging.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


class JobLogger:
    """Per-job logger that writes detailed logs to files."""
    
    def __init__(self, job_id: str, console: Console):
        self.job_id = job_id
        self.console = console
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create job-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"job_{job_id}_{timestamp}.log"
        
        # Setup file logger
        self.logger = logging.getLogger(f"hedra.job.{job_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler with detailed format
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Rich console handler for terminal output
        console_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True
        )
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"🎬 Started job {job_id}")
        self.logger.debug(f"Log file: {self.log_file}")
    
    def log_step(self, step: str, data: Dict[str, Any] = None) -> None:
        """Log a workflow step with optional data."""
        message = f"Step: {step}"
        if data:
            message += f" | Data: {json.dumps(data, indent=2)}"
        self.logger.info(message)
    
    def log_api_request(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> None:
        """Log API request details."""
        self.logger.debug(f"API Request: {method} {endpoint}")
        if params:
            # Sanitize sensitive data
            safe_params = self._sanitize_params(params)
            self.logger.debug(f"Request params: {json.dumps(safe_params, indent=2)}")
    
    def log_api_response(self, response_data: Any, status_code: int = None) -> None:
        """Log API response details."""
        if status_code:
            self.logger.debug(f"API Response Status: {status_code}")
        
        if hasattr(response_data, 'model_dump'):
            data = response_data.model_dump()
        elif hasattr(response_data, '__dict__'):
            data = response_data.__dict__
        else:
            data = response_data
            
        self.logger.debug(f"Response data: {json.dumps(data, indent=2, default=str)}")
    
    def log_asset_upload(self, asset_type: str, file_path: Path, response_url: str) -> None:
        """Log asset upload with file details."""
        file_size = file_path.stat().st_size / 1024  # KB
        self.logger.info(f"📤 Uploaded {asset_type}: {file_path.name} ({file_size:.1f}KB)")
        self.logger.debug(f"Upload URL: {response_url}")
    
    def log_voice_resolution(self, voice_name: str, voice_id: str) -> None:
        """Log voice name to ID resolution."""
        self.logger.info(f"🔍 Resolved voice: {voice_name} → {voice_id}")
    
    def log_bounding_box(self, bbox_str: str, parsed_coords: Dict[str, float]) -> None:
        """Log bounding box parsing and coordinates."""
        self.logger.info(f"🎯 Bounding box: {bbox_str}")
        self.logger.debug(f"Parsed coordinates: {json.dumps(parsed_coords, indent=2)}")
    
    def log_face_detection(self, image_path: Path, faces_count: int, selected_face: Dict[str, Any] = None) -> None:
        """Log face detection results."""
        self.logger.info(f"👥 Detected {faces_count} faces in {image_path.name}")
        if selected_face:
            self.logger.debug(f"Selected face: {json.dumps(selected_face, indent=2)}")
    
    def log_project_status(self, status: str, progress: float = None, video_url: str = None) -> None:
        """Log project status updates."""
        message = f"📊 Status: {status}"
        if progress is not None:
            message += f" ({progress:.1%})"
        self.logger.info(message)
        
        if video_url:
            self.logger.info(f"🎥 Video ready: {video_url}")
    
    def log_error(self, error: Exception, context: str = None) -> None:
        """Log error details with context."""
        if context:
            self.logger.error(f"❌ Error in {context}: {type(error).__name__}: {error}")
        else:
            self.logger.error(f"❌ Error: {type(error).__name__}: {error}")
        
        # Log full traceback to file only
        import traceback
        self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    
    def log_completion(self, success: bool, output_file: Path = None) -> None:
        """Log job completion."""
        if success:
            message = f"✅ Job {self.job_id} completed successfully"
            if output_file:
                message += f" → {output_file}"
            self.logger.info(message)
        else:
            self.logger.error(f"❌ Job {self.job_id} failed")
        
        self.logger.info(f"📋 Full log saved to: {self.log_file}")
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from params for logging."""
        safe_params = params.copy()
        
        # Replace sensitive keys with placeholders
        sensitive_keys = ['api_key', 'token', 'password', 'secret']
        for key in sensitive_keys:
            if key in safe_params:
                safe_params[key] = "[REDACTED]"
        
        # Truncate very long URLs or base64 data
        for key, value in safe_params.items():
            if isinstance(value, str) and len(value) > 200:
                if value.startswith(('http', 'data:')):
                    safe_params[key] = value[:100] + "...[TRUNCATED]"
        
        return safe_params


class EnhancedProgressTracker:
    """Enhanced progress tracker with logging integration."""
    
    def __init__(self, console: Console, job_logger: JobLogger = None):
        self.console = console
        self.job_logger = job_logger
    
    def show_success(self, message: str) -> None:
        """Show success message and log it."""
        self.console.print(f"[green]✅ {message}[/green]")
        if self.job_logger:
            self.job_logger.logger.info(f"✅ {message}")
    
    def show_warning(self, message: str) -> None:
        """Show warning message and log it."""
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")
        if self.job_logger:
            self.job_logger.logger.warning(f"⚠️  {message}")
    
    def show_info(self, message: str) -> None:
        """Show info message and log it."""
        self.console.print(f"[blue]ℹ️  {message}[/blue]")
        if self.job_logger:
            self.job_logger.logger.info(f"ℹ️  {message}")
    
    def show_error(self, message: str) -> None:
        """Show error message and log it."""
        self.console.print(f"[red]❌ {message}[/red]")
        if self.job_logger:
            self.job_logger.logger.error(f"❌ {message}")


def setup_hedra_logging(debug: bool = False) -> None:
    """Setup the main Hedra SDK logging."""
    from hedra._utils._logs import setup_logging
    
    if debug:
        os.environ["HEDRA_LOG"] = "debug"
    else:
        os.environ["HEDRA_LOG"] = "info"
    
    setup_logging()


def create_job_logger(job_id: str, console: Console) -> JobLogger:
    """Create a new job logger instance."""
    return JobLogger(job_id, console)