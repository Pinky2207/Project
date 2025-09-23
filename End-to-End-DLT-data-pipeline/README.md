End-to-End Data Pipeline with dlt, AWS, and Orchestration
📌 Project Overview

This project demonstrates the design and implementation of an end-to-end data pipeline using dlt , a modern data loading framework. The pipeline extracts raw data from external APIs, applies transformations such as schema normalization and type handling, and loads the processed data into multiple destinations including MySQL, AWS S3, and a data warehouse with Apache Iceberg for partitioning.

The project emphasizes scalability, reliability, and automation, incorporating performance tuning techniques and modern orchestration tools like GitHub Actions and Apache Airflow.

image
🚀 Key Features
API Data Extraction
Used dlt resources for secure API ingestion with configuration and secrets management.

Supported both full batch and incremental data loads.

Schema Normalization & Evolution
Automatic schema detection and type conversions.

Flattening nested JSON structures into relational tables.

Implemented data contracts to detect and alert schema changes from source systems.

Data Loading & Storage
Ingested data into inbuilt SQLAlchemy DB, later migrated to MySQL.

Optimized for both batch and incremental updates.

Loaded curated datasets into a data lake (AWS S3 + Glue).

Leveraged Apache Iceberg for partitioning and efficient querying.

Integrated downstream loading into a data warehouse for analytics.

Performance Optimization
Implemented parallelism and asynchronous API requests for faster ingestion.

Applied memory management techniques to handle large-scale datasets.

Orchestration & Deployment
Automated pipeline runs with GitHub Actions (CI/CD).

Scheduled and monitored workflows using Apache Airflow.

🛠️ Tech Stack
Languages & Frameworks: Python, dlt, SQLAlchemy
Databases: MySQL
Cloud & Storage: AWS S3, AWS Glue, Apache Iceberg, PySpark, Redshift
Orchestration & CI/CD: Apache Airflow, GitHub Actions
Other Tools: Pandas, NumPy, Asyncio, Requests
⚡ How It Works
Extraction: Data pulled from APIs with dlt using secure configs/secrets.

Transformation: Schema auto-detection, flattening nested JSON, handling type mismatches, and triggering schema evolution alerts.

Loading: Data ingested into MySQL for operational use, S3 + Glue for data lake storage, and Iceberg-partitioned warehouse for analytics.

Performance Tuning: Applied async requests and parallelism to improve ingestion efficiency.

Orchestration: GitHub Actions handles CI/CD, while Airflow schedules recurring ingestion jobs.

📊 Outcomes
Automated end-to-end ELT pipeline with scalable design.

Reduced manual schema management with data contract + evolution alerts.

Achieved faster ingestion through parallel and async techniques.

Enabled analytics-ready storage via partitioned Iceberg tables in a cloud data lake.

Showcased CI/CD best practices and workflow orchestration for production readiness.

🎯 Skills Demonstrated
Data Engineering: ETL/ELT pipelines, schema evolution, data lake/warehouse integration

Big Data: AWS Glue, S3, Iceberg partitioning

Programming: Python (dlt, Pandas, Asyncio, Requests)

Database: MySQL migration and queries

Performance Engineering: Parallelism, async requests, memory optimization

Orchestration & DevOps: GitHub Actions (CI/CD), Apache Airflow scheduling

Cloud & Scalability: AWS-based ingestion, data lake design
