{% macro generate_schema_name(custom_schema_name, node) -%}
    {#- Keep the warehouse layout explicit and pollution-free:
        - our models use their declared schema directly (silver -> SILVER,
          gold -> GOLD) instead of the default target-schema prefixing;
        - the elementary observability package is isolated into its own
          ELEMENTARY schema so raw BRONZE stays untouched by tooling;
        - under the 'ci' target every schema is prefixed CI_ so a PR build
          can never collide with the dev warehouse layout.
    -#}
    {%- if node.package_name == 'elementary' -%}
        {%- if target.name == 'ci' -%}{{ 'ci_elementary' }}{%- else -%}{{ 'elementary' }}{%- endif -%}
    {%- elif custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {%- if target.name == 'ci' -%}{{ 'ci_' ~ custom_schema_name | trim }}{%- else -%}{{ custom_schema_name | trim }}{%- endif -%}
    {%- endif -%}
{%- endmacro %}
