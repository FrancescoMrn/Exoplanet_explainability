#!/bin/bash
gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region $REGION \
    --scale-tier=BASIC \
    --job-dir gs://${BUCKET_NAME}/jobs/${JOB_NAME} \
    --master-image-uri $IMAGE_URI \
    -- \
	--bucket=${BUCKET_NAME} \
	--output_dir=logs
