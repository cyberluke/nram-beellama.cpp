$base = "http://127.0.0.1:8145"
$plan = '{"model":"brow","prompt":"Analyze this video","media":[{"type":"video","content_id":"sha256:rec-xyz","width":1280,"height":720,"n_frames":600,"fps":30}],"context_length":16384,"max_response_tokens":256}'

Write-Host "=== PLAN 1 (expect cache miss) ==="
$r = Invoke-WebRequest -Uri "$base/v1/beehive/plan" -Method Post -Body $plan -ContentType "application/json" -UseBasicParsing -TimeoutSec 15
$j = $r.Content | ConvertFrom-Json
Write-Host "  decoded_cached=$($j.media[0].decoded_frames_cached)  projected_cached=$($j.media[0].projected_embedding_cached)  tokens=$($j.media[0].tokens_estimate)"

Write-Host "=== CACHE MARK decoded_frame (simulate executor finishing) ==="
$mark = '{"model":"brow","kind":"decoded_frame","content_ids":["sha256:rec-xyz"]}'
$r = Invoke-WebRequest -Uri "$base/v1/beehive/cache" -Method Post -Body $mark -ContentType "application/json" -UseBasicParsing -TimeoutSec 10
$mj = $r.Content | ConvertFrom-Json
Write-Host "  marked=$($mj.marked) refused=$($mj.refused) cache=$($mj.cache.total_entries)"

Write-Host "=== PLAN 2 (expect decoded_frames_cached hit) ==="
$r = Invoke-WebRequest -Uri "$base/v1/beehive/plan" -Method Post -Body $plan -ContentType "application/json" -UseBasicParsing -TimeoutSec 15
$j2 = $r.Content | ConvertFrom-Json
Write-Host "  decoded_cached=$($j2.media[0].decoded_frames_cached)  projected_cached=$($j2.media[0].projected_embedding_cached)"
Write-Host "  preproc: $($j2.preprocessing | ConvertTo-Json -Compress)"

Write-Host "=== CACHE MARK with UNTRUSTED id (expect refused) ==="
$bad = '{"model":"brow","kind":"decoded_frame","content_ids":["foo-bar"]}'
$r = Invoke-WebRequest -Uri "$base/v1/beehive/cache" -Method Post -Body $bad -ContentType "application/json" -UseBasicParsing -TimeoutSec 10
$bj = $r.Content | ConvertFrom-Json
Write-Host "  marked=$($bj.marked) refused=$($bj.refused)"