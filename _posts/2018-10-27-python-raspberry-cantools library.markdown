---
layout: post
title: Raspberry pi with cantools  
date: 2018-10-23 05:20:00
img: python/raspberry/raspberry.png
categories: [python-raspberry] 
tags: [raspberry, cantools] # add tag
---

If you are interested in vehicle and want to get and send `CAN`, you need to some H/W for accessing a car.
There are a few H/W for handling `CAN` message such as Vector, But it's too much expensive to handle individually.
Indeed, If you want to make another application with this message, you need to handle it with program.

We will talk about how to handle the `CAN` with relatively low price H/W and S/W tools.

### H/W : Raspberry Pi

You need H/W Raspberry pi and PICAN2 Board.

+ Raspberry pi

![pi3](../assets/img/python/raspberry/cantools/pi3.jpg)

+ PICAN2 (Image is only 1-channel board. If necessary, you can buy multi-channel board) 

![pican2](../assets/img/python/raspberry/cantools/pican2.jpg)

Assemble Raspberry pi with PICAN2 like example below.

![pican2_pi](../assets/img/python/raspberry/cantools/pican2_pi.jpg)


If you are familiar with Raspberry-pi then, you easily set it up.

### O/S setting: Linux(NOOBS or Raspbian)

We will skip the way to install OS in Raspberry. It's simple and search in Google.
After preparation of initial setting, turn on a terminal.

+ First, update the raspberry

```python
sudo apt-get update 
sudo apt-get upgrade 
sudo reboot
```

<br>

+ Next, add the overlays by typing or using cmd line(ONLY for NOOBS O/S user).

```python
sudo nano /boot/config.txt
```

<br>

if you are not able to access `/boot/config.txt` for authorization, type below.
```python
sudo chmod 777 /home/pi
```

<br>

add the overlays in `/boot/config.txt`.

```python
dtparam=spi=on 
dtoverlay=mcp2515-can0,oscillator=16000000,interrupt=25 
dtoverlay=spi-bcm2835-overlay
```

<br>

![overlays](../assets/img/python/raspberry/cantools/overlays.jpg)

<br>

+ Reboot pi :

```python
sudo reboot
```

<br>

+ You can now bring the CAN interface up:

```python
sudo /sbin/ip link set can0 up type can bitrate 500000
```

<br>

Above, `can0` is correspond to channel and `500000` to bitrate(can-speed, e.g. 500kbps).

That's it. Until now, we end up basic setting. If you reboot your raspberry, Re-type the command in order to set the communication environment.

```python
sudo /sbin/ip link set can0 up type can bitrate 500000
```

<br>

### Python settings

We will use `Python3`. In terminal, we are going to install package `can-tools`. Type below.

```python
sudo apt-get install python3-pip
```

<br>

+ Install

```python
pip install cantools
```

<br>

### How to read `CAN-DB` file

TBD.

### How to transmit `CAN` message

TBD.

### How to receive `CAN` message

TBD.

### Useful code

TBD.


### Reference

[https://cantools.readthedocs.io/en/latest/#cantools](https://cantools.readthedocs.io/en/latest/#cantools)

[https://pypi.org/project/cantools/](https://pypi.org/project/cantools/)

[http://skpang.co.uk/catalog/pican2-canbus-board-for-raspberry-pi-2-p-1475.html](http://skpang.co.uk/catalog/pican2-canbus-board-for-raspberry-pi-2-p-1475.html)






