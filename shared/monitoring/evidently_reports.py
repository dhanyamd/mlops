import os
from datetime import datetime
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from shared.observability.logging import get_logger
from shared.config import DATA_LAKE

log = get_logger(__name__)


class DriftDashboardGenerator:
    """Evidently-based data drift and quality report generator."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.report_dir = os.path.join(DATA_LAKE, "monitoring", project_name)
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_drift_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target_column: str | None = None,
    ) -> str:
        """Create an HTML report containing Data Drift, Target Drift, and Data Quality.
        
        Returns the local path to the generated HTML file.
        """
        log.info(
            "generating_evidently_drift_report",
            project=self.project_name,
            ref_rows=len(reference),
            cur_rows=len(current),
        )
        
        metrics = [
            DataDriftPreset(),
            DataQualityPreset(),
        ]
        
        if target_column and target_column in reference.columns and target_column in current.columns:
            metrics.append(TargetDriftPreset())
            
        report = Report(metrics=metrics)
        report.run(reference_data=reference, current_data=current)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_drift.html"
        output_path = os.path.join(self.report_dir, filename)
        
        report.save_html(output_path)
        log.info("evidently_drift_report_saved_locally", path=output_path)
        
        # Upload to S3 data lake if configured
        try:
            from shared.clients import S3Client
            s3 = S3Client()
            s3_key = f"monitoring/{self.project_name}/{filename}"
            
            # Read local file and upload
            with open(output_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            # Since S3Client has write_df, we can use direct s3fs or boto3
            # S3Client's storage_options are available, let's write using s3fs
            import s3fs
            fs = s3fs.S3FileSystem(**s3.storage_options)
            s3_path = f"s3://{s3.cfg.bucket_name}/{s3_key}"
            with fs.open(s3_path, "w", encoding="utf-8") as s3_f:
                s3_f.write(html_content)
                
            log.info("evidently_drift_report_uploaded_to_s3", s3_path=s3_path)
        except Exception as e:
            log.warning("evidently_drift_report_s3_upload_failed", error=str(e))
            
        return output_path
