---
layout: post
title: How to remote control Raspberry  
date: 2018-10-23 05:20:00
img: python/embedded/raspberry.png
categories: [python-embedded] 
tags: [raspberry, remote control, ssh, vnc, winscp] # add tag
---

We can create server or things with `Raspberry pi`. It's not expensive and its performance is good for simple job.
We need monitor to handle it. But I don't feel good to carry the monitor(e.g. 7-inch raspberry-pi monitor) every time.
In that case, we can remote-access to raspberry with CLI/GUI environment.

<br>

### SSH : CLI environment remote control

SSH stands for Secure Shell Protocol. SSH usually is used for **data transfer** and **remote control**.
In order to remote access to `raspberry pi`, it has to be `enable` in `SSH`.

Click the **Raspberry pi Icon (Upper-Left menu) → Raspberry PI Configuration**. And, Check the SSH as `Enable`.

![remote-control](../assets/img/python/embedded/remote-control/ssh_config.png)

<br>

Next, to remote access, download [Putty](https://www.putty.org/).

![Putty](../assets/img/python/embedded/remote-control/putty.png)

<br>

Put the `Raspberry pi` ip into the Host Name(or IP address) and click the open button.
You can easily verify `Raspberry pi` ip in the terminal with `ifconfig` command.
And unless you have changed the passward, your initial ID : pi, Passward : raspberry. Thus, you type it.

![ssh_connect](../assets/img/python/embedded/remote-control/ssh_connect.png)

<br>

### SSH : data transfer with WinSCP

SSH can not only remote control but also data transfer. First, enable SSH in `Raspberry pi`.
And download [WinSCP](https://winscp.net/eng/download.php). This program supports data transfer from `Window` to `Raspberry pi`.

![winscp_login](../assets/img/python/embedded/remote-control/WinSCP_login.png)

<br>

In the same way, type your ip in host name and connect. That's it!

![winscp_connect](../assets/img/python/embedded/remote-control/WinSCP_connect.png)

<br>

You can transfer data with GUI environment.

<br>

### VNC : GUI environment remote control

It's simpler than SSH method. Download [VNC viewer](https://www.realvnc.com/en/connect/download/viewer/).

like SSH setting, `Raspberry pi` need to be enable VNC.

![vnc](../assets/img/python/embedded/remote-control/vnc_setting.png)

<br>

Run `VNC viewer` you downloaded. 

![vnc_viewer](../assets/img/python/embedded/remote-control/vnc_viewer.png)

<br>

Type `Raspberry pi` Username and Password(pi, raspberry)

![nvc_connect](../assets/img/python/embedded/remote-control/vnc_connet.png)

<br>

### How to set static IP in Raspberry pi ?

Whenever you connect to `Raspberry pi`, you must know `IP address`. But without setting, `Raspberry pi` set IP as automatically.
In other words, you should check the `ip address` when you try to connect. we don't feel like it.
Let's look into the way how to set `static IP address`.

first, get **Gateway** in terminal with `netstat -nr` command.

![gateway](../assets/img/python/embedded/remote-control/gateway.png)

<br>

The address is maybe different mine with yours. Next, revise the `dhcpcd.conf` file.

```python
sudo vi /etc/dhcpcd.conf
```

<br>

In this file, add the static ip and gateway you searched like below.
```python
interface wlan0
static ip_address=192.168.0.10
static routers=192.168.0.1
static domain_name_servers=192.168.0.1
```

<br>

+ interface wlan0 : If you use wifi then, use `wlan0`, if you use LAN, use `eth0`.
+ static ip_address=192.168.0.10 : type what address you want to set
+ static routers=192.168.0.1 : set gateway
+ static domain_name_servers=192.168.0.1 : set gateway

<br>

### Reference

