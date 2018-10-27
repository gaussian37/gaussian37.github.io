---
layout: post
title: How to solve unable to login problem.  
date: 2018-10-23 05:20:00
img: python/raspberry/raspberry.png
categories: [python-raspberry] 
tags: [raspberry, login] # add tag
---

If you have a problem unable to login even though you enter correct ID/Password.
Maybe you are in infinite loops of log in.

![login](../assets/img/python/raspberry/login-problem/login.jpg)

In this case, follow below procedure.

1. Boot your Rsapberry Pi.

2. If you are now in the log-in form, enter Ctrl+Alt+F1(or F2).

3. Log-in with your accounts(normally, user : pi, password : raspberry)

4. You can access the Raspberry-pi then, open a terminal.

5. Type ``` cd ~ ```

6. Type ``` la -A ```  You will now see which files are in your home directory. Look if you see a file called .Xauthority.

7. Type ``` mv .Xauthority .Xauthority.backup ```

8. Type ``` sudo chmod 777 /home/pi ```

9. Type ``` sudo reboot ```

That's it! You solve the problem.