@echo off
setlocal
where tesseract >nul 2>nul
if %errorlevel%==0 (
  echo Tesseract OCR is already available on PATH.
  tesseract --version
  exit /b 0
)
if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" (
  echo Tesseract OCR is installed at %ProgramFiles%\Tesseract-OCR\tesseract.exe
  echo SafeRx will detect this path automatically.
  exit /b 0
)
if exist "%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe" (
  echo Tesseract OCR is installed at %ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe
  echo SafeRx will detect this path automatically.
  exit /b 0
)
echo Tesseract OCR was not found.
echo.
echo If winget is available, run:
echo   winget install --id UB-Mannheim.TesseractOCR -e
echo.
echo Then restart the SafeRx backend.
exit /b 1
