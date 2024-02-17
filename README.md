# Few shot learninig MAML

### Autorzy: Anna Bielawska, [Marcin Kądziołka](https://github.com/MarcinKadziolka)


Celem dzisiejszych zajęć jest poznanie algorytmu MAML (Model Agnostic Meta Learning), a także zrozumienie teorii stojącej za few shot learningiem.
Proponujemy rozpocząć przygodę od notebooka MAML.ipynb, w którym znajduje się wstęp i opis teorii potrzebnej do rozwiązania wszystkich zadań. 
Następnie należy przejść do notebooka CNN_MAML.ipynb, w którym to pobawimy się w trenowanie modelu konwolucyjnego do klasyfikacji few-shot na zbiorach z obrazkami.

Repozytorium zawiera pliki:
- MAML.ipynb -- zawiera wprowadzenie do tematu oraz przykład rozwiązywania problemu regresji logistycznej na samodzielnie generowanym zbiorze,
- CNN_MAML.ipynb -- tutaj stosujemy MAML do uczenia modelu CNN, zadania będą dotyczyły zbiorów MNIST i [Omniglot](https://github.com/brendenlake/omniglot),
- utils.py -- funkcje pomocnocze do notebooka CNN_MAML.ipynb, 
- requirements.txt -- zestaw potrzebnych bibliotek do uruchomienia kodu.

## Konfiguracja środowiska

W trakcie tych ćwiczeń będziemy korzystać z notatników Jupyter Notebook. Aby przygotować środowisko, należy zainstalować niezbędne biblioteki z pliku `requirements.txt`. Zaleca się przygotowanie wcześniej wirtualnego środowiska:

Stwórz środowisko za pomocą `venv`:
```bash
$ python3.9 -m venv .venv
```
lub z użyciem `conda`:
```bash
$ conda create -n .venv python=3.9
```

zainstaluj niezbędne biblioteki:
```bash
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Informacje dodatkowe

Warto zapozać się z artykułem dotyczącym [MAML](https://arxiv.org/pdf/1703.03400.pdf).

## Źródła:

Repozytorium zostało stworzone na podstawie:<br/> 
[1] https://colab.research.google.com/drive/1hqY8rp8aRVm_a7DvIBHH_MHf0c9URQpn?usp=sharing<br/>
[2] https://github.com/cnguyen10/few_shot_meta_learning/tree/master (data dostępu: 07.12.2023)<br/>
[3] https://github.com/brendenlake/omniglot (data dostępu: 07.12.2023)<br/>

Warto zwrócić uwagę na:<br/>
[4] Few Shot Learning basic concepts: https://www.youtube.com/watch?v=hE7eGew4eeg (data dostępu: 07.12.2023)<br/>
[5] Meta Learning: https://www.youtube.com/watch?v=YkeE7oRxF24&t=1s (data dostępu: 07.12.2023)<br/>
[6] Omniglot dataset paper: https://www.science.org/doi/abs/10.1126/science.aab3050<br/>
[7] PyTorch autograd tutorial: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html<br/>