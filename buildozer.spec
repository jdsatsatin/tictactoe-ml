[app]
title = TicTacToe ML
package.name = tictactoeml
package.domain = org.example

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 0.1
requirements = python3,kivy,numpy

[buildozer]
log_level = 2

[app]
android.permissions = INTERNET
orientation = portrait
fullscreen = 0

android.api = 31
android.minapi = 21
android.ndk = 25b
android.sdk = 31
android.accept_sdk_license = True
