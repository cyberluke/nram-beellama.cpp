$base = "http://127.0.0.1:8148"
$body = '{"models":[{"id":"omn","path":"D:\\_MODELS\\LiquidAI\\LFM3-Omni-2B-GGUF\\LFM3-Omni-2B-Q8_0.gguf"},{"id":"brow","path":"D:\\_MODELS\\Duckets\\LFM3-VL-2B-BrowserAgent-GGUF\\lfm3-vl-2b-browseragent-1f00-Q8_0.gguf"},{"id":"abl","path":"D:\\_MODELS\\abliter8-ai\\lfm3-2b-vl-abliter8-gguf\\LFM3-VL-2B-Abliter8-Q8_0.gguf"}],"contextLength":8192}'
Write-Host "=== POST /v1/beehive/assemblies (3 real models, ctx 8192) ==="
$r = Invoke-WebRequest -Uri "$base/v1/beehive/assemblies" -Method Post -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 240
Write-Host "HTTP $($r.StatusCode)"
$j = $r.Content | ConvertFrom-Json
Write-Host "models: $(($j.models | ForEach-Object { "$($_.id)=$($_.status)" }) -join ', ')"
Write-Host "assembly_id: $($j.assembly_id)"
Write-Host "resident_bytes: $($j.optimization.measurements.resident_weight_bytes)"
Write-Host "aliases: $($j.optimization.measurements.total_aliases)  shape_groups: $($j.optimization.measurements.shape_groups)  semantic: $($j.optimization.measurements.semantic_aliases)"
Write-Host "kv measured: $($j.optimization.kv.measured_total_resident_bytes)"
Write-Host "context_capacity: $($j.optimization.context_capacity | ConvertTo-Json -Compress)"