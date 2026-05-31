# GeoMusic

Sistema multimídia interativo que transforma formas geométricas capturadas pela
webcam em comandos musicais.

## O que o projeto faz

- Captura vídeo em tempo real pela webcam.
- Usa como ponto de partida o notebook da aula: extrai características parecidas
  com as do Colab, principalmente área aproximada do contorno e número de lados.
- Treina uma Árvore de Decisão no navegador usando exemplos sintéticos de
  círculo, quadrado e triângulo.
- Classifica a forma detectada e toca uma nota musical diferente.
- Mostra o vídeo da câmera, a caixa de detecção, os medidores das
  características e a sequência musical criada.

## Como rodar localmente

Use um servidor local simples, porque navegadores costumam bloquear webcam em
arquivos abertos diretamente.

```bash
python3 -m http.server 8000
```

Depois acesse:

```text
http://localhost:8000
```

Clique em **Iniciar câmera** e permita o acesso à webcam.

## Como demonstrar no vídeo de entrega

1. Abra o projeto no navegador.
2. Clique em **Iniciar câmera**.
3. Mostre um círculo escuro desenhado em papel claro para tocar **C4**.
4. Mostre um quadrado para tocar **E4**.
5. Mostre um triângulo para tocar **G4**.
6. Mostre a sequência sendo montada na lateral.

Para melhores resultados, use formas escuras em um papel claro, com boa
iluminação. O detector ignora objetos grandes ou pouco geométricos, como rosto,
mãos e sombras, para evitar falsos positivos.
