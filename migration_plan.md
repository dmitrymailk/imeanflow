# Отчет по запуску `imeanflow` без полного ImageNet

## Короткий вывод

Да, этот репозиторий можно запустить без скачивания полного ImageNet и без масштабной переделки кода.

Статус на текущий момент: минимальный локальный `train smoke test` уже успешно выполнен на `RTX 5090`.

Что уже подтверждено практикой:

- локальный `venv` рабочий
- fake-ImageNet из `example_input.jpg` успешно подготовлен
- латенты и `imagenet_256_fid_stats.npz` успешно созданы
- короткий train прогон завершился
- checkpoint сохранен в `/code/auto_remaster/sandbox/imeanflow/_runs/train_smoke/checkpoint_4`
- лоссы печатаются локально в консоль без `wandb`
- короткая FID/IS-оценка тоже доходит до конца

Для вашей машины с `RTX 5090` и без TPU это также означает, что исходный `scripts/install.sh` в старом виде использовать нельзя: он был TPU-ориентирован. Для GPU-ноды зависимости нужно ставить по NVIDIA-сценарию.

Под "запустить" в вашем случае правильно понимать именно минимальный запуск обучения, а не `eval` готового чекпойнта.

Практический путь для этого один:

1. взять одну локальную картинку
2. продублировать ее в маленький фейковый `ImageNet`-подобный набор
3. подготовить латенты штатным кодом репозитория
4. запустить `train` на 1 эпоху с маленькими батчами

Это не reproducing paper. Это именно проверка, что локальный training pipeline этого репозитория реально поднимается: подготовка данных, загрузка латентов, обучение, семплинг, checkpoint и финальная короткая оценка.

Для текущего smoke test в качестве исходной картинки можно использовать уже добавленный файл:

- `/code/auto_remaster/sandbox/imeanflow/example_input.jpg`

## Текущее состояние окружения

Локальное окружение для этого репозитория уже поднято в:

- `/code/auto_remaster/sandbox/imeanflow/.venv`

Базовая проверка окружения уже пройдена успешно.

Проверенные версии внутри `.venv`:

- `jax 0.10.0`
- `jaxlib 0.10.0`
- `flax 0.12.7`
- `orbax-checkpoint 0.11.39`
- `diffusers 0.38.0`
- `torch 2.11.0+cu130`
- `torchvision 0.26.0+cu130`
- `tensorflow 2.21.0`
- `tensorflow_datasets 4.9.10`

Проверка GPU тоже уже успешна:

- `torch.cuda.is_available() == True`
- `torch.cuda.get_device_name(0) == NVIDIA GeForce RTX 5090`
- `jax.devices() == [CudaDevice(id=0)]`

Практический вывод: окружение выглядит рабочим, и дальше можно переходить уже к подготовке fake-датасета и первому train smoke test.

## Что я сверил в коде и статье

### Что говорит статья

Статья и код ориентированы на:

- class-conditional `ImageNet 256x256`
- обучение `from scratch`
- работу в латентном пространстве `VAE`
- латенты размера `32x32x4` для входных картинок `256x256x3`
- оценку через `FID-50K`

Это согласуется с `paper/sections/experiments.tex` и `paper/sections/appendix.tex`.

### Что реально ожидает текущий код

По факту текущий код использует не весь оригинальный `ImageNet`, а достаточно узкую файловую структуру:

- подготовка данных читает изображения через `torchvision.datasets.ImageFolder`
- подготовка латентов сейчас итерирует только по split `train`
- обучение читает не картинки, а `.pt`-файлы латентов из `dataset.root/train`
- вычисление `FID` reference тоже берет только `train`

Из этого следуют важные практические выводы:

1. Папка `val/` для минимального запуска сейчас не обязательна.
2. Названия классов могут быть любыми, не обязательно настоящими `n01440764`.
3. Для smoke test достаточно одного класса, если метка остается в диапазоне `[0, 999]`. При одном каталоге класса это будет label `0`, что допустимо.

### Какой VAE брать для первого запуска

Для первого train smoke test лучше не подменять VAE на `black-forest-labs/FLUX.2-dev`.

Причина не в качестве VAE, а в том, что текущий `imeanflow` код жестко привязан к другому контракту:

- подготовка латентов в [auto_remaster/sandbox/imeanflow/utils/data_util.py](/code/auto_remaster/sandbox/imeanflow/utils/data_util.py) загружает именно `FlaxAutoencoderKL` из `pcuenq/sd-vae-ft-{vae_type}-flax`
- train/eval decode path в [auto_remaster/sandbox/imeanflow/utils/vae_util.py](/code/auto_remaster/sandbox/imeanflow/utils/vae_util.py) снова использует тот же `FlaxAutoencoderKL`
- конфиг модели в [auto_remaster/sandbox/imeanflow/configs/default.py](/code/auto_remaster/sandbox/imeanflow/configs/default.py) зашит под `dataset.image_channels = 4`
- в [auto_remaster/sandbox/imeanflow/utils/vae_util.py](/code/auto_remaster/sandbox/imeanflow/utils/vae_util.py) зашиты конкретные mean/std статистики латентов SD-VAE

Практический вывод:

- `AutoencoderKL.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae", ...)` не является drop-in заменой для этого репозитория
- для такого VAE пришлось бы отдельно переписывать encode path, decode path и нормализацию латентов
- это уже не минимальный запуск репозитория, а отдельная адаптация пайплайна

Поэтому для первого запуска обучения берем штатный VAE путь из репозитория:

- `pcuenq/sd-vae-ft-mse-flax`

А идею с `FLUX.2-dev` VAE имеет смысл рассматривать только как отдельный эксперимент после того, как smoke test на штатной конфигурации уже пройдет.

## Важные расхождения между README и текущим состоянием репозитория

Перед запуском лучше учитывать три момента.

### 1. README указывает на `launch.sh`, но такого файла нет

В README для обучения указан `bash scripts/launch.sh JOB_NAME`, но в текущем репозитории есть только:

- `scripts/train.sh`
- `scripts/eval.sh`

То есть для реального запуска обучения надо ориентироваться на `scripts/train.sh` или сразу на прямой вызов `python3 main.py ...`.

### 2. `scripts/install.sh` ориентирован на TPU

В `scripts/install.sh` первая строка ставит `jax[tpu]==0.4.27`.

Если вы не на TPU, а на обычном Linux с GPU или CPU, безопаснее:

- отдельно поставить подходящий wheel `jax/jaxlib` под вашу среду
- затем поставить остальные зависимости из скрипта

Иначе можно получить неподходящую сборку JAX.

Для вашей текущей машины это особенно важно. Локальная среда сейчас выглядит так:

- `Python 3.11.14`
- `NVIDIA Driver 580.95.05`
- `CUDA Version 13.0`
- локальный `venv` в `/code/auto_remaster/sandbox/imeanflow/.venv`
- `torch 2.11.0+cu130`
- `jax 0.10.0` с `CudaDevice(id=0)`

Практический вывод:

- TPU-сборку JAX ставить не надо
- старые жесткие pin-версии для `torch`, `tensorflow`, `orbax`, `ml-dtypes` и `tensorstore` здесь только мешают
- для JAX на 5090 разумная стартовая точка это `jax[cuda13]`
- для PyTorch здесь подходит актуальный стек `cu130`

В репозитории это лучше чинить через GPU-ориентированный install path, а не вручную повторять TPU-команды.

### 3. Для подготовки данных безопаснее вызывать `prepare_dataset.py` напрямую

В `scripts/prepare_data.sh` флаги передаются в виде `--imagenet_root=\"$IMAGENET_ROOT\"`, то есть со встроенными кавычками.

Чтобы не упереться в проблемы с путями, для первого запуска лучше использовать прямой вызов:

```bash
python3 prepare_dataset.py ...
```

а не shell-обертку `prepare_data.sh`.

## Рекомендуемый путь для вашего случая: минимальный fake-ImageNet и короткий train smoke test

Этот путь лучше соответствует формулировке "мне нужен именно сам запуск данного репозитория".

Идея такая:

1. Берем одну картинку.
2. Дублируем ее несколько раз в структуру, похожую на `ImageNet`.
3. Подготавливаем латенты штатным кодом репозитория.
4. Запускаем обучение на 1 эпоху с очень маленькими настройками.

Это не reproducing paper. Это именно проверка, что весь локальный пайплайн живой.

## Почему не стоит оставлять ровно одну копию файла

Технически одна картинка может хватить для части пайплайна, но для стабильного smoke test лучше сделать не 1 файл, а хотя бы `16` копий одной и той же картинки.

Причины:

- `FID` в этом репозитории считает и covariance, поэтому один референсный пример слишком хрупок
- `Inception Score` режет выборку на `10` частей, поэтому совсем маленькое число сэмплов неудобно
- training `DataLoader` идет с `drop_last=True`, значит датасет не должен быть меньше батча
- визуализация train-сэмплов ожидает как минимум `4` картинки

Иными словами: одна исходная картинка подходит, но лучше продублировать ее в 16 файлов.

## Минимальная файловая структура для fake-ImageNet

Достаточно такой структуры:

```text
fake_imagenet/
└── train/
    └── class0/
        ├── 0000.jpg
        ├── 0001.jpg
        ├── ...
        └── 0015.jpg
```

`val/` для текущего кода не обязателен.

Если хотите оставить структуру ближе к README, можно добавить такую же папку `val/class0/`, но это не требуется для минимального запуска.

## Пошаговый сценарий запуска

### Короткий план выполнения

Если формулировать именно план наших следующих действий, то он такой:

1. Используем `/code/auto_remaster/sandbox/imeanflow/example_input.jpg` как единственный исходный образец.
2. Размножаем его в маленький fake-ImageNet из 16 файлов в одном классе `class0`.
3. Для первого запуска оставляем штатный `sd-vae-ft-mse-flax`, а не `FLUX.2-dev` VAE.
4. Прогоняем `prepare_dataset.py`, чтобы получить латенты и маленький `FID` cache.
5. Не правим основной `configs/train_config.yml` для постоянной работы, а делаем отдельный smoke-конфиг под тестовый запуск.
6. Запускаем обучение на 1 эпоху на одном устройстве.
7. Проверяем, что появились checkpoint, картинки в `workdir/images/` и завершилась финальная оценка.

Первые подготовительные пункты уже фактически закрыты:

1. исходная картинка есть
2. окружение и GPU уже валидированы

Значит следующая реальная рабочая стадия теперь такая:

1. собрать fake-ImageNet
2. прогнать `prepare_dataset.py`
3. сделать `train_smoke_config.yml`
4. сделать первый короткий train запуск

### Шаг 1. Подготовить окружение

Рабочая директория:

```bash
cd /code/auto_remaster/sandbox/imeanflow
```

Дальше два варианта:

- если у вас TPU-окружение, можно идти через `bash scripts/install.sh`
- если обычный Linux GPU/CPU, лучше отдельно поставить подходящий `jax/jaxlib`, а потом доставить остальное

Минимально безопасная мысль здесь простая: не ставить TPU wheel вслепую на не-TPU машине.

Для вашей конкретной машины я бы считал базовым такой путь:

```bash
cd /code/auto_remaster/sandbox/imeanflow
source .venv/bin/activate
IMF_ACCELERATOR=gpu bash scripts/install.sh
```

Почему так:

- это ставит GPU-ветку JAX вместо TPU
- по умолчанию используется `jax[cuda13]`, что соответствует вашему драйверу `580.95.05` и `CUDA 13.0`
- существующий `torch` не будет насильно затираться, если он уже установлен

После установки стоит сразу проверить окружение:

```bash
cd /code/auto_remaster/sandbox/imeanflow
source .venv/bin/activate
python3 - <<'PY'
import jax
import torch
print('jax devices:', jax.devices())
print('torch cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('torch device 0:', torch.cuda.get_device_name(0))
PY
```

### Шаг 2. Создать маленький fake-ImageNet из одной картинки

Пример:

```bash
cd /code/auto_remaster/sandbox/imeanflow
source .venv/bin/activate

SRC_IMG=/code/auto_remaster/sandbox/imeanflow/example_input.jpg
FAKE_ROOT=/code/auto_remaster/sandbox/imeanflow/_smoke/fake_imagenet

mkdir -p "$FAKE_ROOT/train/class0"

for i in $(seq -w 0 15); do
    cp "$SRC_IMG" "$FAKE_ROOT/train/class0/${i}.jpg"
done
```

Если у вас видно несколько GPU и вы не хотите подстраивать batch size под multi-device, для первого запуска проще зафиксировать один девайс:

```bash
export CUDA_VISIBLE_DEVICES=0
```

### Шаг 3. Подготовить латенты и маленький FID cache штатным кодом репозитория

Важно: здесь лучше идти напрямую через `prepare_dataset.py`, а не через `scripts/prepare_data.sh`.

```bash
cd /code/auto_remaster/sandbox/imeanflow
source .venv/bin/activate

FAKE_ROOT=/code/auto_remaster/sandbox/imeanflow/_smoke/fake_imagenet
OUT_ROOT=/code/auto_remaster/sandbox/imeanflow/_smoke/prepared

CUDA_VISIBLE_DEVICES=0 python3 prepare_dataset.py \
    --imagenet_root="$FAKE_ROOT" \
    --output_dir="$OUT_ROOT" \
    --batch_size=4 \
    --vae_type=mse \
    --image_size=256 \
    --compute_latent=True \
    --compute_fid=True \
    --overwrite=False
```

Что должно появиться на выходе:

```text
_smoke/prepared/
├── train/
│   ├── 00000000.pt
│   ├── 00000001.pt
│   ├── ...
│   └── 00000015.pt
└── imagenet_256_fid_stats.npz
```

Это уже тот формат, который обучение реально ожидает.

### Шаг 4. Сделать отдельный smoke-конфиг для обучения

Для первого запуска лучше не переписывать основной `configs/train_config.yml`, а сделать отдельный файл вроде `configs/train_smoke_config.yml`.

Рекомендуемый минимальный набор:

```yaml
dataset:
    root: /code/auto_remaster/sandbox/imeanflow/_smoke/prepared

training:
    log_per_step: 1
    checkpoint_per_epoch: 1
    num_epochs: 1
    batch_size: 4
    sample_per_epoch: 1
    fid_per_epoch: 1

fid:
    device_batch_size: 4
    num_samples: 16
    cache_ref: /code/auto_remaster/sandbox/imeanflow/_smoke/prepared/imagenet_256_fid_stats.npz

load_from: ''
```

Пояснения:

- отдельный smoke-конфиг безопаснее, чем временно менять основной train-конфиг
- `batch_size: 4` нужен для того, чтобы `drop_last=True` не обнулил эпоху
- `device_batch_size: 4` нужен, чтобы train sampling и визуализация не упали на слишком маленьком числе сэмплов
- `num_samples: 16` делает финальную оценку короткой, но все еще технически осмысленной для smoke test

### Шаг 5. Запустить обучение напрямую

```bash
cd /code/auto_remaster/sandbox/imeanflow
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --workdir=/code/auto_remaster/sandbox/imeanflow/_runs/train_smoke \
    --config=configs/load_config.py:train
```

Если пойдем через отдельный smoke-конфиг, то есть два варианта:

1. временно заменить содержимое `configs/train_config.yml` на smoke-настройки
2. добавить отдельный loader entry, чтобы запускать именно `train_smoke`

Для аккуратной работы с репозиторием я бы выбрал второй вариант.

### Что считать успешным результатом

Для smoke test достаточно, чтобы выполнились все этапы без падения:

- конфиг загрузился
- latent dataset прочитался из `_smoke/prepared/train`
- модель собралась
- прошла хотя бы одна эпоха
- сохранился checkpoint
- записалась картинка в `workdir/images/`
- отработала финальная оценка на маленьком `FID` cache

Метрики в таком режиме неинформативны. Это нормально.

## Что важно не перепутать

### 1. `image_size` в подготовке и `image_size` в модели означают разное

При подготовке данных:

- `prepare_dataset.py --image_size=256` означает размер входных картинок

В training config:

- `configs/default.py` использует `dataset.image_size = 32`
- это размер латентов, а не пиксельных изображений

Это не ошибка, а ожидаемое поведение.

### 2. Для train smoke test нужен `FID` cache, даже если метрика вам не важна

В текущей логике train в конце все равно вызывает `fid_evaluator` на последней эпохе.

Значит для запуска без правок кода проще:

- один раз сгенерировать маленький `imagenet_256_fid_stats.npz` на fake-датасете
- прописать его в `fid.cache_ref`
- уменьшить `fid.num_samples`

Так вы не лезете в код и не отключаете оценку вручную.

### 3. Если девайсов больше одного, batch size должен быть согласован с ними

Для train path батч в итоге раскладывается по `jax.local_device_count()`.

Поэтому для первого прогона проще всего:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Иначе нужно следить, чтобы:

- `training.batch_size / jax.process_count()` был кратен `jax.local_device_count()`
- датасет был не меньше этого батча

## Что я бы делал на вашем месте прямо сейчас

Если нужен именно короткий практический маршрут без переписывания кода, я бы шел так:

1. Не использовал бы пока полный ImageNet.
2. Не использовал бы `scripts/prepare_data.sh`.
3. Использовал бы `/code/auto_remaster/sandbox/imeanflow/example_input.jpg`.
4. Создал бы fake-ImageNet из этой картинки, продублированной 16 раз.
5. Прогнал бы `prepare_dataset.py` напрямую.
6. Сделал бы отдельный smoke-конфиг под 1 эпоху и маленький `FID`.
7. Запустил бы `python3 main.py --config=configs/load_config.py:train ...` или отдельный loader entry для smoke-конфига.

Это самый короткий путь проверить реальный запуск всего репозитория с минимальным вмешательством.

## Ограничения этого подхода

- Это не воспроизведение результатов статьи.
- Это не проверка качества модели.
- Это только smoke test пайплайна.
- Метрики на fake-датасете будут бессмысленны.
- Полный режим из статьи все равно требует настоящего `ImageNet 256x256`, больших вычислений и, по словам авторов, изначально тестировался на TPU.

## Итоговая рекомендация

Для вашей цели лучший маршрут такой:

- сначала пройти полный smoke test на fake-ImageNet из одной картинки, размноженной в 16 файлов
- затем, если train smoke test проходит, уже масштабировать датасет и compute budget

Это позволяет проверить запуск репозитория сейчас, не скачивая полный `ImageNet` и не переписывая кодовую базу.