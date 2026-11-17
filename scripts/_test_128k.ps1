$base = "http://127.0.0.1:8146"
# load omn at maximum supported context (128000, the model default since n_ctx=0 -> train ctx)
$load = '{"models":[{"id":"omn","path":"D:\\_MODELS\\LiquidAI\\LFM3-Omni-2B-GGUF\\LFM3-Omni-2B-Q8_0.gguf"}],"context_length":0}'
Write-Host "=== load omn (context_length 0 = model max 128K) ==="
$r = Invoke-WebRequest -Uri "$base/v1/beehive/assemblies" -Method Post -Body $load -ContentType "application/json" -UseBasicParsing -TimeoutSec 240
Write-Host "  HTTP $($r.StatusCode)"
Start-Sleep -Seconds 2
$r = Invoke-WebRequest -Uri "$base/v1/beehive/lab-report" -UseBasicParsing -TimeoutSec 15
$j = $r.Content | ConvertFrom-Json
Write-Host "  context_capacity: $($j.optimization.context_capacity | ConvertTo-Json -Compress)"
Write-Host "  kv: $($j.optimization.kv | ConvertTo-Json -Compress)"