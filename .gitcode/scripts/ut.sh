#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set +e
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)

if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
    echo "Switched to gcc-16"
elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
    echo "Switched to gcc-15"
elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
    echo "gcc-16/15 not available, fell back to gcc-14"
fi

if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d /home/jenkins/opensource/gcc15 ]; then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
    elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
    fi
elif gcc --version | head -n1 | grep -q "14\."; then
    gcc --version
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi
if [ "${target_branch}" == "master" ] || [ "${target_branch}" == "develop" ]; then
	sudo update-alternatives --set lcov /opt/lcov-2.3.2/bin/lcov
	lcov --version
fi
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
pip3 install -r super_kernel/requirements-dev.txt

touch test1.log
coverage_save="true"
if [ "${target_branch}" == "master" ] || [ "${target_branch}" == "develop" ]; then
	case "${ut_type}" in
		UT_Test_Python_superkernel)
			bash build.sh -u -c --impl=py --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource" -j20 -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		UT_Test_Cpp_superkernel)
			bash build.sh -u -c --impl=cpp --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		UT_Test_autofuse_framework)
			bash build.sh -u -c --module=autofuse_framework --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		UT_Test_autofuse_ascendc_api)
			bash build.sh -u -c --module=autofuse_ascendc_api --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		ST_Test_Python_superkernel)
			bash build.sh -s -c --impl=py --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		ST_Test_Cpp_superkernel)
			bash build.sh -s -c --impl=cpp --module=superkernel --no-autofuse --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		ST_Test_autofuse_framework)
			bash build.sh -s -c --module=autofuse_framework --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			coverage_save="true"
			;;
		ST_Test_autofuse_ascendc_api)
			bash build.sh -s -c --module=autofuse_ascendc_api --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt -j20
			ret=$?
			coverage_save="true"
			;;
		ST_Test_autofuse_e2e)
			bash build.sh -s -c  --module=autofuse_e2e --cann_3rd_lib_path="/home/jenkins/opensource" -j20 -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			coverage_save="true"
			;;
        *)
            echo "Skip UT test execution for ${ut_type} on non-master branch"
            exit 0
            ;;
	esac
else
	case "${ut_type}" in
		UT_Test_Python_superkernel)
			bash build.sh -u -c --impl=py --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		UT_Test_Cpp_superkernel)
			bash build.sh -u -c --impl=cpp --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		UT_Test_autofuse_framework)
			bash build.sh -u -c --module=autofuse_framework --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		UT_Test_autofuse_ascendc_api)
			bash build.sh -u -c --module=autofuse_ascendc_api --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		ST_Test_Python_superkernel)
			bash build.sh -s -c --impl=py --module=superkernel --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			;;
		ST_Test_Cpp_superkernel)
			bash build.sh -s -c --impl=cpp --module=superkernel --no-autofuse --cann_3rd_lib_path="/home/jenkins/opensource" -f ${WORKSPACE}/pr_filelist.txt
			ret=$?
			;;
		ST_Test_autofuse_framework)
			bash build.sh -s -c --module=autofuse_framework --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			coverage_save="false"
			;;
		ST_Test_autofuse_ascendc_api)
			bash build.sh -s -c --module=autofuse_ascendc_api --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			coverage_save="false"
			;;
		ST_Test_autofuse_e2e)
			bash build.sh -s -c  --module=autofuse_e2e --cann_3rd_lib_path="/home/jenkins/opensource"
			ret=$?
			coverage_save="false"
			;;
        *)
            echo "Skip UT test execution for ${ut_type} on non-master branch"
            exit 0
            ;;
	esac
fi

/usr/local/ccache/bin/ccache -s
if [ $ret -ne 200 ] && [ $ret -ne 0 ]; then
    echo "run ut fail"
    exit 1
fi
if [ $ret -eq 0 ]; then
    if [ "$coverage_save" = "true" ];then
    echo "ut_process=coverage" >> $ATOMGIT_OUTPUT
    else
    echo "ut_process=ut_cov" >> $ATOMGIT_OUTPUT
    fi
fi
exit 0
