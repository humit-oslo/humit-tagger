# humit-tagger

Dette prosjektet tagger teksten som gis som parameter.
For å installere, kreves et python-miljø. Python 3 (testet på python3.8) anbefales.

    ./setup.sh

For å kjøre en prøvetagging kan du enten gi et filnavn som en parameter til tag.py-skriptet ved å bruke -f-alternativet som følgende:

    python3 tag.py -f norwegian_text.txt

For å taggere ved å bruke spesifikt språk

    python3 tag.py -bm -f norwegian_text.txt

eller

    python3 tag.py -nn -f norwegian_text.txt

Bruk -bm for bookmål and -nn for nynorsk.

CUDA-basert GPU anbefales. For å stille inn batchstørrelsen, bruk følgende:

    python3 tag.py -b 16 -f norwegian_text.txt

For å taggere alle filene i en directory, bruk følgende:

    python3 tag.py -i <input_directory> -o <output_directory>

Directory-moden kjøres ikke rekursivt. Filene i output\_directory blir overskrevet.

Skriptet vil returnere json.

## Den nye versjonen

Vi lager en ny versjon som bruker NorBertv3 som basismodell. 
Denne versjonen krever ikke en separat ordsammenslåingsmodell fordi tokeniseringen gir informasjon om en token er sammen med den forrige tokenen eller ikke (inkludert tegnsetting).

For å installere denne versjonen, etter å klone dette repoet:

    ./setup.sh

Da kan du bruke tag\_new.py som du bruker tag.py


# License

[MIT license](https://github.com/textlab/norwegian_ml_tagger/blob/master/LICENSE)


