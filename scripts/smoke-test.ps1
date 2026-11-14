# OpenAI-compatible gateway smoke test (SGLang).
#  1. fetch /models -> take the FIRST model id -> cache it in scripts/.model-cache.json
#     (subsequent runs reuse the cached id; -Refresh forces re-fetch)
#  2. chat completion for the prompt with the native qwen driver semantics
#     (reasoning_content + text content are read straight from the parts)
#  3. PASS only with finish_reason=stop, non-empty content and a real usage tuple.
param(
	[switch]$Refresh
)

$ErrorActionPreference = "Stop"
$base = "http://13.218.50.224:30000/v1"
$key = "banikpico1337"
$cachePath = Join-Path $PSScriptRoot ".model-cache.json"
$headers = @{ Authorization = "Bearer $key" }

$models = $null
$modelId = $null
if (-not $Refresh -and (Test-Path $cachePath)) {
	$cached = Get-Content $cachePath -Raw | ConvertFrom-Json
	$modelId = $cached.modelId
	Write-Output "[cache] model=$modelId (from $($cached.fetchedAt))"
	$models = @($modelId)
}
if (-not $modelId) {
	$r = Invoke-RestMethod -Uri "$base/models" -Headers $headers
	$ids = @($r.data | ForEach-Object { $_.id })
	if ($ids.Count -lt 1) { Write-Error "no models listed"; exit 1 }
	$modelId = $ids[0]
	Write-Output "[models] $($ids -join ', ') -> first: $modelId"
	@{ modelId = $modelId; fetchedAt = (Get-Date).ToString("o") } | ConvertTo-Json | Set-Content $cachePath
}

$sw = [Diagnostics.Stopwatch]::StartNew()
$body = @{
	model    = $modelId
	messages = @(@{ role = "user"; content = "analyze tech stack" })
} | ConvertTo-Json -Depth 5
$resp = Invoke-RestMethod -Uri "$base/chat/completions" -Method Post -Headers $headers -ContentType "application/json" -Body $body
$sw.Stop()

$choice = $resp.choices[0]
$content = $choice.message.content
$reasoning = $choice.message.reasoning_content
$finish = $choice.finish_reason
$usage = $resp.usage
Write-Output "[timing] $($([Math]::Round($sw.Elapsed.TotalSeconds,1))) s  finish=$finish  in=$($usage.prompt_tokens) out=$($usage.completion_tokens) reasoning=$($usage.completion_tokens_details.reasoning_tokens 2>$null)"
if ($reasoning) { Write-Output "[reasoning] $($reasoning.Substring(0, [Math]::Min(200, $reasoning.Length)))" }
Write-Output "[content]"
Write-Output $content

if (-not $content -or $finish -ne "stop") {
	Write-Error "SMOKE FAIL: content=$([bool]$content) finish=$finish"
	exit 1
}
Write-Output "SMOKE PASS"
