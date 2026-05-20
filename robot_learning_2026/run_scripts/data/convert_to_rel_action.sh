lerobot-edit-dataset \
    --repo_id "DerBoroter/full_fold_tristan" \
    --new_repo_id "DerBoroter/full_fold_tristan_rel_action" \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size 50 \
    --push_to_hub true