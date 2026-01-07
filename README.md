# Radio telescope software for HI line detection

Developed for the International Olympiad on Astronomy and Astrophysics (IOAA)
Primary Development: Shirish Pathare - HBCSE, TIFR

Based on original code by: Ashish Mhaske - IUCAA, Pune

With guidance from:
    Prof. Avinash Deshpande, RRI, Bengaluru and Prof. Surhud More, IUCAA, Pune

Group task problem design
    Prof. Avinash Deshpande, RRI, Bengaluru, Prof. Dharam Vir Lal, NCRA-TIFR, Pune, and Prof. Amol Dighe, TIFR, Mumbai 

With thanks to: IOAA Academic Committee

## Installation

### System requirements

Blacklist unwanted rtl drivers

```
# sudo echo blacklist dvb_core > /etc/modprobe.d/blacklist-rtlsdr.conf
# sudo echo blacklist dvb_usb_rtl2832u >> /etc/modprobe.d/blacklist-rtlsdr.conf
# sudo echo blacklist dvb_usb_rtl28xxu >> /etc/modprobe.d/blacklist-rtlsdr.conf
# sudo echo blacklist dvb_usb_v2 >> /etc/modprobe.d/blacklist-rtlsdr.conf
# sudo echo blacklist r820t >> /etc/modprobe.d/blacklist-rtlsdr.conf
# sudo echo blacklist rtl2830 >> /etc/modprobe.d/blacklist-rtlsdr.conf
# sudo echo blacklist rtl2832 >> /etc/modprobe.d/blacklist-rtlsdr.conf
# sudo echo blacklist rtl2832_sdr >> /etc/modprobe.d/blacklist-rtlsdr.conf
# sudo echo blacklist rtl2838 >> /etc/modprobe.d/blacklist-rtlsdr.conf
```

```
# sudo modprobe -r dvb_core
# sudo modprobe -r dvb_usb_rtl28xxu
# sudo modprobe -r dvb_usb_v2
# sudo modprobe -r r820t
# sudo modprobe -r rtl2830
# sudo modprobe -r rtl2832
# sudo modprobe -r rtl2832_sdr
```

#### For Ubuntu:
```
# sudo apt-get install librtlsdr-dev
# sudo apt-get install rtl-sdr
# sudo apt install libxcb-cursor0
```


#### For Fedora:
```
# sudo dnf install rtl-sdr rtl-sdr-devel
# sudo dnf install xcb-util-cursor xcb-util-cursor-devel
```

### Conda installation

```
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

# source /̃miniconda3/bin/activate
# conda update -n base -c defaults conda

# conda create -n ioaa
# conda activate ioaa
# conda create -n ioaa python=3.8
```

Pip install requirements
```
pip install -r requirements.txt
```

### Normal installation

If you already have python>=3.8, then you may be able to just install from requirements.txt below.

Pip install requirements
```
pip install -r requirements.txt
```

## Launch program

```
# rtl_biast -b 1
# python ioaa25_gc.py
```

## Documentation

- [Group-Write-Up.pdf](https://ioaa2025.in/wp-content/uploads/2025/09/Gropu-Write-Up.pdf)
- [Group-Design-Horn-Antenna.pdf](https://ioaa2025.in/wp-content/uploads/2025/09/Group-Design-Horn-Antenna.pdf)
- [Group-Design-Antenna-Stand.pdf](https://ioaa2025.in/wp-content/uploads/2025/09/Group-Design-Horn-Antenna-Stand.pdf)
- [Group-Info-Electronics.pdf](https://ioaa2025.in/wp-content/uploads/2025/09/Group-Info-Electronics.pdf)

- [Group-Questions.pdf](https://ioaa2025.in/wp-content/uploads/2025/09/Group-Questions.pdf)
- [Group-Summary-Answersheet.pdf](https://ioaa2025.in/wp-content/uploads/2025/09/Group-Summary-Answersheet.pdf)
- [Group-Solutions.pdf](https://ioaa2025.in/wp-content/uploads/2025/09/Group-Solutions.pdf)

## Bugs

If you find bugs or any issues, please see [here](https://github.com/surhudm/IOAA2025_RadioAntenna/issues) if it is already reported, and if not add a new issue.
