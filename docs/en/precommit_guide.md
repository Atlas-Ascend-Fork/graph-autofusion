# pre-commit Usage Guide

English | [简体中文](../zh/precommit_guide.md)

[TOC]
--

## 1 Background

This guide mainly describes how to use the pre-commit capabilities deployed in the code repository locally (mainly including code formatting and OAT scanning capabilities).

## 2 Feature Overview

1. After installing pre-commit, code formatting processing and OAT checks will be automatically performed before git commits.

2. Compliance issues will block commits and prompt for modifications. Blocking is not forced - you can ignore the modifications.

## 3 Community Contributors Using pre-commit Capabilities

### 3.1 pre-commit Installation Steps

Step 1: Install pre-commit framework

```bash
# Using pip (recommended)
pip install pre-commit

# Verify installation
pre-commit --version
# Output: pre-commit 3.x.x
```

**Windows Users**: Make sure Python and pip are installed.

Step 2: Enter project directory

```bash
cd /path/to/your/project

# For example
cd d:\complianceRepo\CANN
```

Step 3: Install Git Hooks

```bash
# Run in project root directory
pre-commit install
```

Step 4: Verify installation (optional)

```bash
# Test hook (won't actually commit)
git commit --allow-empty -m "test pre-commit"
```

Subsequently, code formatting processing and OAT checks will be automatically performed before committing code.

### 3.2 OAT Usage Guide

**OAT (Open Source Audit Tool)** is an open source compliance checking tool, automatically integrated into the Git commit workflow.

#### 3.2.1 Check Content

**File Type Check** - Binary files (.so, .dll, .exe, etc.) are prohibited from being submitted
**License Header Check** - Verifies source code files contain compliant license declarations

#### 3.2.2 Core Features

-  **Incremental Check** - Only checks files to be committed, fast (< 5 seconds)
-  **Automatic Trigger** - Runs automatically on every `git commit`
-  **Detailed Reports** - Automatically generates `result.txt` summary and full report
-  **Zero Configuration** - Java and Maven are automatically installed (Linux/macOS)
-  **Cross-Platform** - Full support for Windows/Linux/macOS

#### 3.2.3 Required Software

| Software | Version Requirement | Purpose | Installation Method |
|------|---------|------|----------|
| **Java** | JRE 8+ | Run OAT |  **Auto-install** (Linux/macOS)<br> Manual install (Windows)|
| **Maven** | 3.5+ | Package OAT |  **Auto-install** (Linux/macOS)<br> Manual install (Windows)|
| **Git** | 2.0+ | Version Control | Usually already installed |
| **pre-commit** | 2.0+ | Hook Framework | `pip install pre-commit` |

#### 3.2.4 Auto-Installation Support

| Platform | Java | Maven | Package Manager | First Install Time |
|------|------|-------|---------|-------------|
| **Linux (Ubuntu/Debian)** | Auto |  Auto | apt | 5-8 minutes |
| **Linux (CentOS/RHEL)** |  Auto |  Auto | yum | 5-8 minutes |
| **macOS** |  Auto |  Auto | Homebrew | 8-10 minutes |
| **Windows** |  Manual |  Manual | - | Requires manual install |

#### 3.2.5 Important Note: Auto-Skip on Environment Issues

**Friendly Design**: If Java/Maven cannot be installed or environment issues are encountered, OAT check will **automatically skip**, and commit will continue.

**Scenarios That Will Auto-Skip**

| Scenario | Behavior | Prompt |
|------|------|------|
| Java/Maven not installed (Windows) |  Skip check, allow commit | Provides manual installation guide |
| Java/Maven auto-install fails |  Skip check, allow commit | Prompts manual installation method |
| Maven packaging fails |  Skip check, allow commit | Provides solution |
| OAT scan execution fails |  Skip check, allow commit | Prompts to repackage |

**Scenarios That Will Still Block Commits**

| Scenario | Behavior | Reason |
|------|------|------|
| **Binary files found** |  Block commit | Real compliance issue |
| **License header missing/incorrect** | Block commit | Real compliance issue |

**Skip Check Prompt Example**

```
[OAT] Windows cannot auto-install Java
[OAT] Please manually download and install:
  ... (installation steps) ...

[OAT] Skipping OAT check, continuing commit...
[OAT] Recommend installing Java and running check again
```

**Manually Run Check Later**

After configuring the environment, you can manually run the check:

```bash
# Recommended method
pre-commit run oat-check

# Or run script directly
bash scripts/oat_check.sh
```

#### 3.2.6 Compliance Issues (Block Commit)

**Important**: The following issues will **block commits** and must be fixed.

**1) Invalid File Type Found**

**Scenario**: Attempting to commit binary files (.so, .dll, .exe, etc.).

**Output**:
```
====================================================================
  Compliance Issues Found
====================================================================

[OAT] Found 1 compliance issue(s):
  - Invalid File Type: 1
  - License Header Invalid: 0

[OAT] Details saved to: oat_reports/single/result.txt
[OAT] Please check the report and fix the issues.

To view the summary:
  cat oat_reports/single/result.txt

To skip this check temporarily:
  git commit --no-verify
```

**Behavior:** **Blocks commit, must fix**

**View Details**:
```bash
cat oat_reports/single/result.txt
```

**Report Content Example**:
```
===================================
OAT Scan Result Summary
===================================
Scan Time: 2026-03-25 14:30:15
Project: CANN
Files Checked: 1

-----------------------------------
Invalid File Type Total Count: 1
lib/libtest.so: BINARY_FILE_TYPE

-----------------------------------
License Header Invalid Total Count: 0

===================================
Full report: oat_reports/single/PlainReport_CANN.txt
===================================
```

**Solution**:
```bash
# Method 1: Remove binary file
git reset HEAD lib/libtest.so

# Method 2: Add binary files to .gitignore
echo "*.so" >> .gitignore
echo "*.dll" >> .gitignore
echo "*.exe" >> .gitignore

# Re-commit
git add .gitignore
git commit -m "update: add binary files to gitignore"
```

**2) Invalid License Header**

**Scenario**: Source code file is missing or has incorrect license header format.

**Output**:
```
====================================================================
  Compliance Issues Found
====================================================================

[OAT] Found 2 compliance issue(s):
  - Invalid File Type: 0
  - License Header Invalid: 2

[OAT] Details saved to: oat_reports/single/result.txt
```

**Behavior**: **Blocks commit, must fix**

**View Details**:
```bash
cat oat_reports/single/result.txt
```

**Report Content Example**:
```
===================================
OAT Scan Result Summary
===================================

-----------------------------------
Invalid File Type Total Count: 0

-----------------------------------
License Header Invalid Total Count: 2
src/main.cpp: MISSING_LICENSE_HEADER
src/utils.cpp: MISSING_LICENSE_HEADER

===================================
```

**Solution**:

Add license header at the top of the file, for example CANN-2.0:

```cpp
/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

```

**Re-commit**:
```bash
git add src/main.cpp src/utils.cpp
git commit -m "fix: add license headers"
```

---

#### 3.2.7 Report Viewing

**Report File Locations**

| Report Type | File Path | Content |
|---------|---------|------|
| **Summary Report** | `oat_reports/single/result.txt` | Key issue summary  |

**View Commands**

```bash
# View report
cat oat_reports/single/result.txt

# View with editor
code oat_reports/single/result.txt
vim oat_reports/single/result.txt
```

**Summary Report Content**

```
===================================
OAT Scan Result Summary
===================================
Scan Time: 2026-03-25 14:30:15
Project: CANN
Files Checked: 3

-----------------------------------
Invalid File Type Total Count: 0

-----------------------------------
License Header Invalid Total Count: 0

===================================
Full report: oat_reports/single/PlainReport_CANN.txt
===================================
```

#### 3.2.8 Environment Issues

**1) Java Not Installed (Linux/macOS Auto-Install)**

**Scenario**: First commit, Java not installed on system.

**Output**:
```
====================================================================
  Java Not Installed - Attempting Auto-Install
====================================================================

[OAT] Detected Java not installed, starting auto-install...
[OAT] Installing OpenJDK 11 using apt...
[OAT] [OK] OpenJDK 11 installed successfully
```

**Handling**: Auto-install, may require sudo password.

---

**2) Java Not Installed (Windows Manual Install)**

**Scenario**: Windows system cannot auto-install Java.

**Output**:
```
[OAT] Windows cannot auto-install Java
[OAT] Please manually download and install:

  1. Visit: https://adoptium.net/
  2. Download: Eclipse Temurin JRE 11 (x64)
  3. Restart Git Bash after installation
  4. Verify: java -version

[OAT] Skipping OAT check, continuing commit...
[OAT] Recommend installing Java and running check again
```

**Behavior**: **Skip check, allow commit**

**Follow-up Actions**:
1. Install Java manually as prompted
2. Restart terminal
3. Run `pre-commit run oat-check` to verify environment

---

**3)  Java Auto-Install Fails**

**Scenario**: Java auto-install fails on Linux/macOS.

**Output**:
```
[OAT] [ERROR] Auto-install failed

[OAT] Auto-install failed, skipping OAT check

Manual installation method:
  Linux:   sudo apt install openjdk-11-jre
  macOS:   brew install openjdk@11
  Windows: https://adoptium.net/

[OAT] Continuing commit (compliance check not performed)...
[OAT] Recommend installing Java and running: pre-commit run oat-check
```

**Behavior**: **Skip check, allow commit**

**Possible Reasons**:
- Network connection issues
- Package manager not configured
- Insufficient permissions
- Homebrew not installed on macOS

**Solution**:
```bash
# Linux
sudo apt update
sudo apt install openjdk-11-jre

# macOS - Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install openjdk@11

# Verify
java -version

# Manually run check
pre-commit run oat-check
```

---

**4) Maven Not Installed (Linux/macOS Auto-Install)**

**Scenario**: First commit, Maven not installed on system.

**Output**:
```
====================================================================
  Maven Not Installed - Attempting Auto-Install
====================================================================

[OAT] Installing Maven using apt...
[OAT] [OK] Maven installed successfully
```

**Handling**: Auto-install, may require sudo password.

---

**5) Maven Not Installed (Windows Manual Install)**

**Scenario**: Windows system cannot auto-install Maven.

**Output**:
```
[OAT] Windows cannot auto-install Maven
[OAT] Please manually download and install:

  1. Visit: https://maven.apache.org/download.cgi
  2. Download: apache-maven-3.x.x-bin.zip
  3. Extract to C:\Program Files\apache-maven-3.x.x
  4. Add to system PATH
  5. Restart Git Bash
  6. Verify: mvn -version

[OAT] Skipping OAT check, continuing commit...
[OAT] Recommend installing Maven and running check again
```

**Behavior**: **Skip check, allow commit**

**Follow-up Actions**: Install Maven manually as prompted, then run `pre-commit run oat-check`

---

**6) Maven Packaging Fails**

**Scenario**: Maven fails to package OAT JAR.

**Output**:
```
====================================================================
  Maven Packaging Failed
====================================================================

[OAT] Cannot package OAT JAR, skipping OAT check

Possible reasons:
  1. Maven configuration issues
  2. Network connection issues (cannot download dependencies)
  3. pom.xml configuration errors

Suggested solutions:
  1. Manual packaging:
     cd ../tools_oat
     mvn clean package -DskipTests

  2. Configure Maven mirror (China network):
     Edit ~/.m2/settings.xml to add Aliyun mirror

[OAT] Continuing commit (compliance check not performed)...
[OAT] Recommend fixing packaging issues and running: pre-commit run oat-check
```

**Behavior**:  **Skip check, allow commit**

**Solution**:

**Method 1: Manual Packaging**
```bash
cd ../tools_oat
mvn clean package -DskipTests

# View output, should see BUILD SUCCESS
```

**Method 2: Configure Aliyun Mirror (China Network)**
```bash
mkdir -p ~/.m2
cat > ~/.m2/settings.xml <<'EOF'
<settings>
  <mirrors>
    <mirror>
      <id>aliyun</id>
      <mirrorOf>central</mirrorOf>
      <name>Aliyun Maven Mirror</name>
      <url>https://maven.aliyun.com/repository/public</url>
    </mirror>
  </mirrors>
</settings>
EOF

# Re-package
cd ../tools_oat
mvn clean package -DskipTests
```

**Method 3: Get JAR from Team**
```bash
# If team already has compiled JAR, copy directly
# Copy JAR file to ../tools_oat/target/ directory
```

**Verify Fix**:
```bash
pre-commit run oat-check
```

---

**7) tools_oat Clone Fails**

**Output**:
```
[OAT] tools_oat not found. Cloning...
[OAT] [ERROR] Failed to clone tools_oat.
[OAT] You can manually clone from: https://gitcode.com/openharmony-sig/tools_oat.git
```

**Reason**: Network connection issues.

**Solution**:
```bash
# Method 1: Check network
ping gitcode.com

# Method 2: Manual clone
cd ..
git clone https://gitcode.com/openharmony-sig/tools_oat.git

# Method 3: Configure proxy
git config --global http.proxy http://proxy.example.com:8080

# Method 4: Copy from team member
# Have a colleague who already cloned package the tools_oat folder for you
```

---

**8) OAT Scan Execution Fails**

**Scenario**: OAT JAR fails to run.

**Output**:
```
====================================================================
  OAT Scan Execution Failed
====================================================================

[OAT] Scan failed, skipping OAT check

Possible reasons:
  1. JAR file corrupted
  2. Java version incompatible
  3. OAT configuration issues

Suggested solutions:
  1. Delete and re-package JAR:
     rm ../tools_oat/target/ohos_ossaudittool-*.jar
     cd ../tools_oat && mvn clean package -DskipTests

  2. Check Java version (requires Java 8+):
     java -version

[OAT] Continuing commit (compliance check not performed)...
[OAT] Recommend fixing scan issues and running: pre-commit run oat-check
```

**Behavior**: **Skip check, allow commit**

**Solution**:
```bash
# Step 1: Delete old JAR
rm ../tools_oat/target/ohos_ossaudittool-*.jar

# Step 2: Re-package
cd ../tools_oat
mvn clean package -DskipTests

# Step 3: Verify JAR
ls -lh target/ohos_ossaudittool-*.jar

# Step 4: Run check
cd -
pre-commit run oat-check
```
