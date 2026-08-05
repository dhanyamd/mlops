{% macro generate_schema_name(custom_schema_name, node) -%}
    {#- Keep the warehouse layout explicit and pollution-free:
        - our models use their declared schema directly (silver -> SILVER,
          gold -> GOLD) instead of the default target-schema prefixing;
        - the elementary observability package is isolated into its own
          ELEMENTARY schema so raw BRONZE stays untouched by tooling.
    -#}
    {%- if node.package_name == 'elementary' -%}
        {{ 'elementary' }}
    {%- elif custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
