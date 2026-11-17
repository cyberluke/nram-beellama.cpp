# beehive_concurrent_probe.ps1 — fire N concurrent image-description probes at an
# OpenAI-compatible server, verify all complete without crash/freeze, and measure
# wall-time + aggregate throughput. Reusable concurrency load-test.
param(
  [int]$Port = 43901,
  [string]$ImagePath = "C:\Users\lukes.COREI9\Downloads\854046627_4203758799759626_4150600114181312093_n.jpg",
  [int]$N = 6
)
$ProgressPreference = "SilentlyContinue"
$enc = [Convert]::ToBase64String([IO.File]::ReadAllBytes($ImagePath))
$sw = [Diagnostics.Stopwatch]::StartNew()

$probe = {
  param($port,$enc,$i)
  try {
    $body = @{ model="LFM3-VL-2B-Abliter8"; messages=@(@{role="user"; content=@(
      @{type="image_url"; image_url=@{url="data:image/jpeg;base64,$enc"}},
      @{type="text"; text="Describe this image. Reply very briefly."}
    )}); max_tokens=80 } | ConvertTo-Json -Depth 10 -Compress
    $r = Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 180
    return [pscustomobject]@{ i=$i; ok=$true; fin=$r.choices[0].finish_reason; toks=$r.usage.completion_tokens; pts=$r.usage.prompt_tokens }
  } catch {
    return [pscustomobject]@{ i=$i; ok=$false; fin="ERR"; toks=0; pts=0; err=$_.Exception.Message }
  }
}

$jobs = @()
for ($i = 1; $i -le $N; $i++) {
    $jobs += Start-Job -ScriptBlock $probe -ArgumentList $Port, $enc, $i
}
$res = $jobs | Wait-Job -Timeout 200 | Receive-Job
$jobs | Remove-Job -Force
$sw.Stop()

$ok = ($res | Where-Object { $_.ok }).Count
$fail = ($res | Where-Object { -not $_.ok }).Count
$totalToks = ($res | Measure-Object -Property toks -Sum).Sum
"=== $N concurrent image probes on :$Port ==="
"completed ok = $ok, failed = $fail, in $([math]::Round($sw.Elapsed.TotalSeconds,1))s"
if($totalToks -gt 0){ "aggregate $([math]::Round($totalToks / $sw.Elapsed.TotalSeconds,1)) tok/s wall" }
# server aliveness after
$health = try { (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5).status } catch { "DOWN" }
"server after = $health"
# latency spread
"per-request tokens: $((($res | ForEach-Object { $_.toks }) -join ','))"
if($fail -gt 0){ "failures detail:"; $res | Where-Object { -not $_.ok } | ForEach-Object { "  request $($_.i): $($_.err)" } }
exit 0