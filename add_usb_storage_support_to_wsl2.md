# WSL2增加U盘访问支持

由于wsl未将usb存储设备支持编入内核，在WSL2中增加U盘访问支持需要重新编译内核。

1. 相关软件包安装

   首先要安装编译和配置内核所需要的相关软件包

   ```
   $ sudo apt install build-essential flex bison dwarves libssl-dev libelf-dev bc ncurses-dev
   ```

2. 下载并解压内核

    从https://github.com/microsoft/WSL2-Linux-Kernel/releases 选择一个合适版本的内核下载，并解压。

   ```
   $ wget https://github.com/microsoft/WSL2-Linux-Kernel/archive/refs/tags/linux-msft-wsl-5.15.90.1.tar.gz
   $ tar xzf linux-msft-wsl-5.15.90.1.tar.gz
   ```

3. 配置内核

   ```
   $ cd WSL2-Linux-Kernel-linux-msft-wsl-5.15.90.1
   $ zcat /proc/config.gz >.config
   $ make menuconfig
   ```

   在内核配置菜单中选择`Device Drivers  --->[*] USB support  ---> <*>   USB Mass Storage support` 

4. 编译内核

   ```
   $ make bzImage
   ```

5. 将内核拷贝到windows系统

   在windows中以管理员模式打开一个PowerShell，在c盘创建一个目录，例如kernel，将x86/boot/bzImage文件拷贝到该目录

   ```
   mkdir c:\kernel
   cp \\wsl.localhost\Ubuntu-22.04\home\user\WSL2-Linux-Kernel-linux-msft-wsl-5.15.90.1\arch\x86\boot\bzImage C:\kernel\
   ```

6. 修改wsl配置

   用powershell在用户目录下创建一个 WSL 2的配置文件.wslconfig

   ```
   cd $env:USERPROFILE
   echo "[wsl2]">.wslconfig
   echo "kernel=C:\\kernel\\bzImage" >>.wslconfig
   ```

7. 重新启动wsl

   在powershell中重启wsl

   ```
   wsl --shutdown
   ```

   等待10秒左右，运行下面的命令查看wsl是否全都关闭

   ```
   wsl --list --running
   ```

   若看到提示“没有正在运行的分发。”，则说明wsl已全部关闭

8. 检查内核的编译时间

   重新打开wsl后，运行下面的命令检查确认内核的编译时间

   ```
   $ uname -v
   ```

   若输出的内核版本的编译时间与自己编译的内核时间一致，则说明内核已更改成功

9. 安装 USBIPD-WIN

   在powershell中运行下面的命令

   ```
   winget install --interactive --exact dorssel.usbipd-win
   ```

10. wsl 中安装 USBIP 工具和硬件数据库

    ```
    sudo apt install linux-tools-generic hwdata
    sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/*-generic/usbip 20
    ```

11. 查看可连接的usb设备

    在windows中以管理员模式打开一个新的PowerShell，列出所有连接到 Windows 的 USB 设备：

    ````
    usbipd wsl list
    ```
    ````

12. 将U盘连接到wsl

    根据`usbipd wsl list`的输出，找到U盘所对应的USB大容量存储设备的BUSID，连接到wsl

    ```
    usbipd wsl attach --busid <busid>
    ```

13. 检查确认U盘连接成功

    在wsl中运行lsusb确认已连接

    ```
    $ lsusb
    ```

    若已连上，可以用dmesg查看确认u盘设备是否已识别

    ```
    $ dmesg|grep -i usb -A 8
    ```

    另外也可以用lsblk查看连接的磁盘，进行确认

    ```
    lsblk -S|grep usb
    ```

    若有输出，则说明已完成u盘连接

14. 断开u盘

    使用完后，可以在powershell中用以下命令断开u盘

    ```
    usbipd wsl detach --busid <busid>
    ```
