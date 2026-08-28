"""Hality -- triagem de indicio de halitose a partir de foto da lingua.

Nao e diagnostico. Ver README.md e docs/ARQUITETURA.md.
"""
# HEIC e o formato padrao de foto no iPhone e aparece no conjunto proprio.
# Registrado aqui, no import do pacote, para valer em treino e em inferencia --
# registrar so no pipeline deixava o treino quebrando em S32.HEIC.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - ambiente sem o pacote ainda funciona p/ jpg/png
    pass
