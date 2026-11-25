# Security Statement

## Running User Recommendations

For security reasons, it is not recommended to use administrator accounts such as root to execute any commands. Follow the principle of least privilege.

## File Permission Control

- It is recommended that users set the running system umask value to 0027 or higher on the host (including the host machine) and in containers. This ensures that new folders have a default maximum permission of 750 and new files have a default maximum permission of 640.
- It is recommended that users implement security measures such as permission control for sensitive content such as personal privacy data, business assets, source files, and various files saved during the development process. For example, for project installation directory permission control and input public data file permission control, see [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario).
- Users need to implement permission control during installation and use. For file permission settings, see [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario).

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. During the compilation process, some intermediate files will be generated. It is recommended that you implement permission control for intermediate files after compilation to ensure file security.

## Running Security Statement

When an exception occurs during runtime, the process will exit and print an error message. It is recommended to locate the specific error cause based on the error prompt.

## Public Network Address Statement

The public network addresses contained in this project code are as follows:

|      Type      |                                           Open Source Code Address                                           |                            File Name                             |             Public Network IP Address/Public Network URL Address/Domain Name/Email Address/Compressed File Address             | Purpose Description                        |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:----------------------------|
|  Dependency  | Not applicable  | docs/zh/build.md、docs/en/build.md | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz | Download makeself source code from gitcode as compilation dependency |
|  Dependency  | Not applicable  | super_kernel/requirements-dev.txt | https://pypi.tuna.tsinghua.edu.cn/simple | Download python whl packages from pypi as compilation and runtime dependencies |
|  Dependency  | Not applicable  | cmake/fetch_cann_cmake.cmake | https://gitcode.com/cann/cmake.git | Download CANN cmake as compilation dependency |


## Vulnerability Mechanism Description
[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario

| Type           | Linux Permission Reference Maximum Value |
| -------------- | ---------------  |
| User home directory                        |   750 (rwxr-x---)            |
| Program files (including script files, library files, etc.)       |   550 (r-xr-x---)             |
| Program file directory                      |   550 (r-xr-x---)            |
| Configuration file                          |  640 (rw-r-----)             |
| Configuration file directory                      |   750 (rwxr-x---)            |
| Log files (recording completed or archived)        |  440 (r--r-----)             |
| Log files (currently recording)                |    640 (rw-r-----)           |
| Log file directory                      |   750 (rwxr-x---)            |
| Debug files                         |  640 (rw-r-----)         |
| Debug file directory                     |   750 (rwxr-x---)  |
| Temporary file directory                      |   750 (rwxr-x---)   |
| Maintenance upgrade file directory                  |   770 (rwxrwx---)    |
| Business data files                      |   640 (rw-r-----)    |
| Business data file directory                  |   750 (rwxr-x---)      |
| Key components, private keys, certificates, ciphertext file directories    |  700 (rwx------)      |
| Key components, private keys, certificates, encrypted ciphertext        | 600 (rw-------)      |
| Encryption/decryption interfaces, encryption/decryption scripts            |   500 (r-x------)        |
