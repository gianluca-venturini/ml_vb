#!/bin/bash
sudo pip install virtualenv
virtualenv .
source ./bin/activate
pip install tensorflow
pip install selenium
pip install Pillow
# Download geckodriver for Mac
curl -L https://github.com/mozilla/geckodriver/releases/download/v0.18.0/geckodriver-v0.18.0-macos.tar.gz | tar xz -C bin
# Download Firefox for Mac
curl http://download.cdn.mozilla.net/pub/mozilla.org/firefox/releases/20.0/mac/en-US/Firefox%2020.0.dmg > /tmp/Firefox.dmg
hdiutil mount /tmp/Firefox.dmg
cp -R "/Volumes/Firefox/Firefox.app" bin/
hdiutil unmount "/Volumes/Firefox"
# Install npm packages
cd image_generator
npm install
cd ..
