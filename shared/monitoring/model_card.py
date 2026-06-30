import os
from datetime import datetime
from jinja2 import Template
from shared.observability.logging import get_logger
from shared.config import DATA_LAKE

log = get_logger(__name__)

TEMPLATE = """# Model Card — {{ model_name }}

## Model Details
- **Developer**: MLOps Team
- **Model Date**: {{ date }}
- **Model Version**: {{ model_version }}
- **Model Type**: {{ model_type }}
- **Framework/Libraries**: {{ framework }}

## Intended Use
- **Primary Intended Uses**: {{ intended_use }}
- **Out-of-Scope Uses**: {{ out_of_scope }}

## Training Data & Features
- **Dataset Source**: {{ training_data_source }}
- **Dataset Size**: {{ dataset_size }} rows
- **Input Features**: 
{% for feature in features %}
  - `{{ feature }}`
{% endfor %}

## Metrics & Performance
We evaluate our champion models using standard metrics on a holdout test set.
{% for metric_name, value in metrics.items() %}
- **{{ metric_name }}**: {{ value }}
{% endfor %}

## Global Explainability (SHAP)
Top contributing features sorted by global SHAP values:
{% for feature in top_features %}
- `{{ feature }}`
{% endfor %}

## Limitations & Considerations
- Retraining schedule: Checked for feature drift on each run.
- Edge cases: Out-of-distribution feature values may cause unpredictable prediction scores.
"""


class ModelCardGenerator:
    """Auto-generates structured Model Cards (markdown) from model training metadata."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.output_dir = os.path.join(DATA_LAKE, "model_cards", project_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_card(self, metadata: dict) -> str:
        """Render the Jinja2 template and write to model_card.md.
        
        Returns the local path of the generated markdown file.
        """
        log.info("generating_model_card", project=self.project_name)
        
        # Add date if not provided
        if "date" not in metadata:
            metadata["date"] = datetime.utcnow().strftime("%Y-%m-%d")
            
        template = Template(TEMPLATE)
        rendered = template.render(**metadata)
        
        output_path = os.path.join(self.output_dir, "model_card.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)
            
        log.info("model_card_saved_locally", path=output_path)
        
        # Upload to S3 if configured
        try:
            from shared.clients import S3Client
            s3 = S3Client()
            s3_key = f"model_cards/{self.project_name}/model_card.md"
            
            import s3fs
            fs = s3fs.S3FileSystem(**s3.storage_options)
            s3_path = f"s3://{s3.cfg.bucket_name}/{s3_key}"
            with fs.open(s3_path, "w", encoding="utf-8") as s3_f:
                s3_f.write(rendered)
                
            log.info("model_card_uploaded_to_s3", s3_path=s3_path)
        except Exception as e:
            log.warning("model_card_s3_upload_failed", error=str(e))
            
        return output_path
