## Generate the SSh Key on PC

Run the following command on the PC terminal

```python
 ssh-keygen -t ed25519 -C "harrinriza93@gmail.com"
```

It will ask you to ask for the password, just press enter.  
and this will create a key under   `~/.ssh/id_ed25519.pub`
So copy the content in the file by using following command.


```python
  cat ~/.ssh/id_ed25519.pub
```

 In github
 

 ```python
  Settings --> SSH and GPG Keys ---> click new SSH Key.
```

Paste your key there and then Press add
