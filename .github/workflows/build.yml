name: Build Android APK

on:
  push:
    branches: [ main, master ]

jobs:
  build:
    runs-on: ubuntu-22.04

    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Build APK with Buildozer
      uses: Artemis-Software/buildozer-action@v1.2
      with:
        command: buildozer -v android debug
        subdirectory: .

    - name: Upload APK Artifact
      uses: actions/upload-artifact@v3
      with:
        name: MemeApp-APK
        path: bin/*.apk
