@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--disable-safe-unpickle --gradio-img2img-tool color-sketch --xformers 

call webui.bat
