$base = "http://127.0.0.1:8143"
$plan = '{"model":"brow","prompt":"x","media":[{"type":"video","content_id":"sha256:vid-abc","width":1280,"height":720,"n_frames":10,"fps":10}],"context_length":4096}'
Write-Host "=== plan 1 (cache miss expected) ==="
$r = Invoke-WebRequest -Uri "$base/v1/beehive/plan" -Method Post -Body $plan -ContentType "application/json" -UseBasicParsing -TimeoutSec 15
$j = $r.Content | ConvertFrom-Json
Write-Host "media[0]: $($j.media[0] | ConvertTo-Json -Compress)"
Write-Host "preproc: $($j.preprocessing | ConvertTo-Json -Compress)"