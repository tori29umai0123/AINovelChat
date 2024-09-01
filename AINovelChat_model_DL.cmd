@echo off
setlocal enabledelayedexpansion

REM Set the base path for models
set "dpath=%~dp0"
echo Base directory set to: %dpath%

goto :main

:download_and_verify
set "FILE_PATH=%~1"
set "DOWNLOAD_URL=%~2"
set "EXPECTED_HASH=%~3"
set "MAX_ATTEMPTS=3"

for /L %%i in (1,1,%MAX_ATTEMPTS%) do (
    echo Attempt %%i of %MAX_ATTEMPTS%
    curl -L "!DOWNLOAD_URL!" -o "!FILE_PATH!"
    
    REM Calculate SHA-256 hash
    for /f "skip=1 tokens=* delims=" %%# in ('certutil -hashfile "!FILE_PATH!" SHA256') do (
        set "ACTUAL_HASH=%%#"
        goto :hash_calculated
    )
    :hash_calculated
    
    if "!ACTUAL_HASH!"=="!EXPECTED_HASH!" (
        echo Hash verification successful.
        exit /b 0
    ) else (
        echo Hash mismatch. Retrying...
        if %%i equ %MAX_ATTEMPTS% (
            echo Warning: Failed to download file with matching hash after %MAX_ATTEMPTS% attempts.
            exit /b 1
        )
    )
)
exit /b

:verify_hash
set "FILE_PATH=%~1"
set "EXPECTED_HASH=%~2"

for /f "skip=1 tokens=* delims=" %%# in ('certutil -hashfile "%FILE_PATH%" SHA256') do (
    set "ACTUAL_HASH=%%#"
    goto :hash_calculated_verify
)
:hash_calculated_verify

if "%ACTUAL_HASH%"=="%EXPECTED_HASH%" (
    echo Hash verification successful for %FILE_PATH%
    exit /b 0
) else (
    echo Hash mismatch for %FILE_PATH%
    echo Expected: %EXPECTED_HASH%
    echo Actual:   %ACTUAL_HASH%
    exit /b 1
)

:download_files_custom
if "%~1"=="" (
    echo No arguments provided to download_files_custom
    exit /b 1
)
echo Downloading files to "%~1" from "%~2" with custom path "%~4"
set "MODEL_DIR=%dpath%\%~1"
set "MODEL_ID=%~2"
set "FILES=%~3"
set "CUSTOM_PATH=%~4"
echo MODEL_DIR: !MODEL_DIR!
echo MODEL_ID: !MODEL_ID!
echo FILES: !FILES!
echo CUSTOM_PATH: !CUSTOM_PATH!

if not exist "!MODEL_DIR!" (
    echo Creating directory !MODEL_DIR!
    mkdir "!MODEL_DIR!"
)

for %%f in (%FILES%) do (
    set "FILE_PATH=!MODEL_DIR!\%%f"
    set "EXPECTED_HASH=!%~1_%%~nf_hash!"
    set "RETRY_COUNT=0"
    :retry_download_custom
    if not exist "!FILE_PATH!" (
        echo Downloading %%f...
        curl -L "https://huggingface.co/!MODEL_ID!/resolve/main/!CUSTOM_PATH!/%%f" -o "!FILE_PATH!"
        if !errorlevel! neq 0 (
            echo Error downloading %%f
        ) else (
            echo Downloaded %%f
            call :verify_hash "!FILE_PATH!" "!EXPECTED_HASH!"
            if !errorlevel! neq 0 (
                echo Hash verification failed for %%f
                set /a RETRY_COUNT+=1
                if !RETRY_COUNT! lss 3 (
                    echo Retry !RETRY_COUNT!/3
                    del "!FILE_PATH!"
                    goto :retry_download_custom
                ) else (
                    echo Hash verification failed after 3 retries. Deleting %%f
                    del "!FILE_PATH!"
                )
            )
        )
    ) else (
        echo %%f already exists. Verifying hash...
        call :verify_hash "!FILE_PATH!" "!EXPECTED_HASH!"
        if !errorlevel! neq 0 (
            echo Hash verification failed for existing file %%f
            del "!FILE_PATH!"
            set "RETRY_COUNT=0"
            goto :retry_download_custom
        )
    )
)
exit /b 0

:download_files_default
if "%~1"=="" (
    echo No arguments provided to download_files_default
    exit /b 1
)
set "MODEL_DIR=%dpath%\%~1"
set "MODEL_ID=%~2"
set "FILES=%~3"

echo MODEL_DIR: !MODEL_DIR!
echo MODEL_ID: !MODEL_ID!
echo FILES: !FILES!

if not exist "!MODEL_DIR!" (
    echo Creating directory !MODEL_DIR!
    mkdir "!MODEL_DIR!"
)

for %%f in (%FILES%) do (
    set "FILE_PATH=!MODEL_DIR!\%%f"
    set "EXPECTED_HASH=!%~1_%%~nf_hash!"
    set "RETRY_COUNT=0"
    :retry_download_default
    if not exist "!FILE_PATH!" (
        echo Downloading %%f...
        curl -L "https://huggingface.co/!MODEL_ID!/resolve/main/%%f" -o "!FILE_PATH!"
        if !errorlevel! neq 0 (
            echo Error downloading %%f
        ) else (
            echo Downloaded %%f
            call :verify_hash "!FILE_PATH!" "!EXPECTED_HASH!"
            if !errorlevel! neq 0 (
                echo Hash verification failed for %%f
                set /a RETRY_COUNT+=1
                if !RETRY_COUNT! lss 3 (
                    echo Retry !RETRY_COUNT!/3
                    del "!FILE_PATH!"
                    goto :retry_download_default
                ) else (
                    echo Hash verification failed after 3 retries. Deleting %%f
                    del "!FILE_PATH!"
                )
            )
        )
    ) else (
        echo %%f already exists. Verifying hash...
        call :verify_hash "!FILE_PATH!" "!EXPECTED_HASH!"
        if !errorlevel! neq 0 (
            echo Hash verification failed for existing file %%f
            del "!FILE_PATH!"
            set "RETRY_COUNT=0"
            goto :retry_download_default
        )
    )
)
exit /b 0

:main
echo Starting main execution

REM Define hashes
set "models_EZO-Common-9B-gemma-2-it.Q8_0_hash=42c46c773e56cedc82232da12d8d1a7da5fc459a8bc21b34ab624544a5c3fa91"
set "models_Mistral-Nemo-Instruct-2407-Q8_0_hash=08aba63fd29ea8fa2f846566e353f78acce2eec1bd6043df7a4242046a0d7ae4"

echo Downloading gguf model:
call :download_files_default "models" "MCZK/EZO-Common-9B-gemma-2-it-GGUF" "EZO-Common-9B-gemma-2-it.Q8_0.gguf"
call :download_files_default "models" "bartowski/Mistral-Nemo-Instruct-2407-GGUF" "Mistral-Nemo-Instruct-2407-Q8_0.gguf"

echo Script execution completed
echo Press Enter to close the script...
pause > nul
exit /b
endlocal