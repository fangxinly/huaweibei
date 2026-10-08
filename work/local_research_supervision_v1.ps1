param([switch]$NewNodeInformation)
$ErrorActionPreference = 'Stop'
if ($NewNodeInformation) { throw 'New nodes require actual recovery checks, not the unchanged-state routine.' }
$taskPaths = @('outputs/研究接续状态.md','outputs/主任务反事实效用实验分析.md','outputs/主任务反事实效用监督边界与后续设计.md')
$taskReadings = foreach ($taskPath in $taskPaths) {
    $taskText = Get-Content -LiteralPath $taskPath -Raw -Encoding UTF8
    [pscustomobject]@{ Path=$taskPath; FirstParagraph=($taskText -split '\r?\n')[2]; SHA256=(Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash }
}
$taskDisks = @(Get-PSDrive -Name C,D | Select-Object Name,Free)
$taskAt = [DateTime]::UtcNow.ToString('o')
$taskRecord = [pscustomobject]@{
    CheckedAt=$taskAt; Readings=$taskReadings; FreeBytes=$taskDisks
    Action='LOCAL_STATUS_READ_ONLY_NO_NEW_NODE_INFORMATION'
    RemoteConnectionAttempted=$false; TrainingStarted=$false; TorchExecuted=$false
    Limits='File readings and fresh local disk only; no GPU or new weight verification.'
}
$taskSaveDirectory = 'D:/CodexBackups/selective_flow_20261003_1105/local_supervision_latest'
New-Item -Path $taskSaveDirectory -ItemType Directory -Force | Out-Null
$taskCheckJson = $taskRecord | ConvertTo-Json -Depth 5
$taskCheckJson | Set-Content -LiteralPath (Join-Path $taskSaveDirectory 'heartbeat_local_status_latest.json') -Encoding UTF8
$taskCheckJson | Set-Content -LiteralPath 'work/heartbeat_local_status_latest.json' -Encoding UTF8
$taskFreeC = ($taskDisks | Where-Object Name -eq C).Free
$taskFreeD = ($taskDisks | Where-Object Name -eq D).Free
$taskStatePath = 'outputs/研究接续状态.md'
$taskState = Get-Content -LiteralPath $taskStatePath -Raw -Encoding UTF8
$taskState = [regex]::Replace($taskState,'(?m)^最近本地监管核对UTC.*\r?\n?','')
$taskLine = '最近本地监管核对UTC' + $taskAt + '：无新有效节点或C最终材料，未重连/训练；证据work/heartbeat_local_status_latest.json。C实查' + $taskFreeC + 'bytes，D' + $taskFreeD + 'bytes，新增大材料继续优先D，无删除。本条仅文件状态和本地空间核对，不是GPU/权重核验。'
$taskNewState = $taskState.TrimEnd() + [Environment]::NewLine + [Environment]::NewLine + $taskLine + [Environment]::NewLine
$taskNewState | Set-Content -LiteralPath (Join-Path $taskSaveDirectory '研究接续状态.md') -Encoding UTF8
$taskNewState | Set-Content -LiteralPath $taskStatePath -Encoding UTF8
$taskSaveRows = foreach ($taskSource in @($taskStatePath,'work/heartbeat_local_status_latest.json','work/local_research_supervision_v1.ps1')) {
    $taskDestination = Join-Path $taskSaveDirectory (Split-Path -Leaf $taskSource)
    if ($taskSource -eq 'work/local_research_supervision_v1.ps1') { Copy-Item -LiteralPath $taskSource -Destination $taskDestination -Force }
    $taskSourceHash = (Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash
    $taskDestinationHash = (Get-FileHash -LiteralPath $taskDestination -Algorithm SHA256).Hash
    if ($taskSourceHash -ne $taskDestinationHash) { throw 'Local supervision save hash mismatch' }
    [pscustomobject]@{ Source=$taskSource; Destination=$taskDestination; SHA256=$taskSourceHash }
}
$taskProof = [pscustomobject]@{ Status='MUTABLE_LATEST_LOCAL_SUPERVISION_THREE_FILES_SHA_VERIFIED'; At=[DateTime]::UtcNow.ToString('o'); Rows=$taskSaveRows; RemoteVerification=$false }
$taskProofJson = $taskProof | ConvertTo-Json -Depth 5
$taskProofJson | Set-Content -LiteralPath (Join-Path $taskSaveDirectory 'heartbeat_local_save_latest.json') -Encoding UTF8
$taskProofJson | Set-Content -LiteralPath 'work/heartbeat_local_save_latest.json' -Encoding UTF8
[pscustomobject]@{ Status=$taskProof.Status; CheckedAt=$taskAt; FreeBytes=$taskDisks; ResearchStateChanged=$false } | ConvertTo-Json -Depth 4
