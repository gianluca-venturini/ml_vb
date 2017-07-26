import * as React from "react";
import * as $ from "jquery";
import * as ReactDOM from "react-dom";
import * as seedrandom from "seedrandom";
import * as queryString from "query-string";

import { Sentence } from "./sentence";

interface State {
  words: string[];
  counter: number;
}

const parameters = queryString.parse(document.location.search);
const WORDS = parseInt(parameters['words']);
const VERTICAL_OFFSET = parseInt(parameters['vertical_offset']);
const HORIZONTAL_OFFSET = parseInt(parameters['horizontal_offset']);
const WIDTH = parseInt(parameters['width']);
const HEIGHT = parseInt(parameters['height']);
const LINE_HEIGHT = parseInt(parameters['line_height']);
const SENTENCES = parseInt(parameters['sentences']);
const KEEP_LINES = parseInt(parameters['keep_lines']);
const SKIP_LINES = parseInt(parameters['skip_lines']);
const LINES = Math.ceil(HEIGHT / (LINE_HEIGHT + VERTICAL_OFFSET));
const BACKGROUND_IMAGES = JSON.parse(parameters['images']) as string[];
const IMAGES_PATH = parameters['images_path'] as string;
const MONOCROME_BACKGROUND = parameters['monocrome_background'] === '1';
const LINE_SAME_FONT = parameters['line_same_font'] === '1';
const MAX_FONT_SIZE = parameters['max_font_size'];

const fontSizes = [
  10,
  10,
  10,
  10,
  11,
  11,
  11,
  11,
  11,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  12,
  14,
  14,
  14,
  14,
  14,
  14,
  14,
  14,
  14,
  15,
  15,
  15,
  15,
  15,
  15,
  15,
  15,
  16,
  16,
  16,
  16,
  16,
  16,
  17,
  20,
  22,
  24,
  25,
  26,
  31,
  39,
  45,
].filter(size => size <= MAX_FONT_SIZE);

const fontWeights: (200 | 400 | 600 | 800)[] = [
  200,
  200,
  200,
  200,
  200,
  200,
  200,
  200,
  200,
  400,
  400,
  400,
  400,
  400,
  400,
  600,
  800
];

const random = seedrandom(`123`);


class Main extends React.Component<{}, State> {

  constructor(props: {}) {
    super(props);

    this.state = {
      words: [],
      counter: 0,
    };
  }

  handleClick = () => {
    this.setState(s => {
      s.counter++;
      return s;
    })
  }

  componentDidMount() {
    $.get('words.txt', data => {
      this.setState(s => {
        s.words = data.split('\n');
        return s;
      })
    });
  }

  generateRandomColor() {
    return Math.floor(random() * 150);
  }

  generateRandomFontSize() {
    return fontSizes[Math.floor(random() * fontSizes.length)]
  }

  generateRandomFontWeight() {
    return fontWeights[Math.floor(random() * fontWeights.length)]
  }

  render() {

    const lineElements = [];
    const imgSrc = `${IMAGES_PATH}/${BACKGROUND_IMAGES[Math.floor(BACKGROUND_IMAGES.length * random())]}`;

    for (let line = 0; line < LINES; line++) {
      const sentences = [];
      let color = this.generateRandomColor();
      let fontSize = this.generateRandomFontSize();
      let fontWeight = this.generateRandomFontWeight();
      for (let sentence = 0; sentence < SENTENCES; sentence++) {
        const words = [];
        /** Randomly select WORDS words from the dictionary */
        for (let j = 0; j < WORDS; j++) {
          words.push(
            this.state.words[Math.round(random() * this.state.words.length)] || ''
          );
        }
        if (!LINE_SAME_FONT) {
          color = this.generateRandomColor();
          fontSize = this.generateRandomFontSize();
          fontWeight = this.generateRandomFontWeight();
        }
        const opacity = ( line % (KEEP_LINES + SKIP_LINES) >= KEEP_LINES) ? 0 : 1;
        sentences.push(
          <Sentence
            color={`rgb(${color}, ${color}, ${color})`}
            horizontalOffset={ HORIZONTAL_OFFSET }
            verticalOffset={ VERTICAL_OFFSET }
            lineHeight={ LINE_HEIGHT }
            opacity={ opacity }
            fontSize={ fontSize }
            fontWeight={ fontWeight }
          >
            { words.join(' ') }
          </Sentence>
        )
      }
      lineElements.push(
        <div
          style={{
            position: 'absolute',
            top: line * LINE_HEIGHT,
          }}
        >
          { sentences }
        </div>
      );
    }

    const style: any = {
      height: '100%',
      width: '100%',
      cursor: 'pointer',
      whiteSpace: 'nowrap',
      font: 'Open Sans',
    };

    if (MONOCROME_BACKGROUND) {
      const color = Math.floor(255 - random() * 10);
      style.backgroundColor = `rgb(${color}, ${color}, ${color})`;
    } else {
      style.backgroundImage = `url("${imgSrc}")`;
      style.backgroundPosition = `${random() * 2000}px ${random() * 2000}px`;
      style.backgroundRepeat = 'repeat';
      style.backgroundSize = '1679px 944px';
    }

    return (
      <div
        id={this.state.words.length > 0 ? "main" : undefined}
        style={style}
        onClick={this.handleClick}
      >
        { lineElements }
      </div>
    );
  }
}

export function mountNode() {
  const mountNode = document.getElementById('root');
  ReactDOM.render(<Main />, mountNode);
}
