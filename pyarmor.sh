pip3 install pyarmor==7.2.0
wget https://airesources.oss-cn-hangzhou.aliyuncs.com/zl/pyarmor.zip
mv pyarmor.zip /root/
unzip /root/pyarmor.zip -d /root

mkdir source/
cp ./* ./source/

pyarmor obfuscate --obf-code 0 --src="./source" -r --output=./ wsgi.py
rm -rf source /root/pyarmor.zip /root/.pyarmor Dockerfile.deploy