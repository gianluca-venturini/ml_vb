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

const fontSizes = [
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
]

const fontWeights: (200 | 400 | 600 | 800)[] = [
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

const parameters = queryString.parse(document.location.search);
const WORDS = parseInt(parameters['words']);
const VERTICAL_OFFSET = parseInt(parameters['vertical_offset']);
const HORIZONTAL_OFFSET = parseInt(parameters['horizontal_offset']);
const WIDTH = parseInt(parameters['width']);
const HEIGHT = parseInt(parameters['height']);
const LINE_HEIGHT = parseInt(parameters['line_height']);
const SENTENCES = parseInt(parameters['sentences']);
const SKIP_LINE = parseInt(parameters['skip_line']);
const SKIP_LINES = parseInt(parameters['skip_lines']);
const LINES = Math.ceil(HEIGHT / (LINE_HEIGHT + VERTICAL_OFFSET));
const BACKGROUND_IMAGES = JSON.parse(parameters['images']) as string[];
const IMAGES_PATH = parameters['images_path'] as string;

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

  render() {

    const sentences = [];
    const imgSrc = `${IMAGES_PATH}/${BACKGROUND_IMAGES[Math.floor(BACKGROUND_IMAGES.length * random())]}`

    for (let line = 0; line < LINES; line++) {
      for (let sentence = 0; sentence < SENTENCES; sentence++) {
        const words = [];
        /** Randomly select WORDS words from the dictionary */
        for (let j = 0; j < WORDS; j++) {
          words.push(
            this.state.words[Math.round(random() * this.state.words.length)] || ''
          );
        }
        const color = Math.floor(random() * 256);
        const opacity = ( line % (SKIP_LINE + SKIP_LINES) >= SKIP_LINE ) ? 0 : 1;
        sentences.push(
          <Sentence
            color={`rgb(${color}, ${color}, ${color})`}
            horizontalOffset={ HORIZONTAL_OFFSET }
            verticalOffset={ VERTICAL_OFFSET }
            lineHeight={ LINE_HEIGHT }
            opacity={ opacity }
            fontSize={ fontSizes[Math.floor(random() * fontSizes.length)] }
            fontWeight={ fontWeights[Math.floor(random() * fontWeights.length)] }
          >
            { words.join(' ') }
          </Sentence>
        )
      }
      sentences.push(<br />);
    }

    return (
      <div
        id={this.state.words.length > 0 ? "main" : undefined}
        style={{
          height: '100%',
          width: '100%',
          cursor: 'pointer',
          whiteSpace: 'nowrap',
          backgroundImage: `url("${imgSrc}")`,
          backgroundPosition: `${random() * 2000}px ${random() * 2000}px`,
          backgroundRepeat: 'repeat',
          backgroundSize: '1679px 944px',
        }}
        onClick={this.handleClick}
      >
        { sentences }
      </div>
    );
  }
}

export function mountNode() {
  const mountNode = document.getElementById('root');
  ReactDOM.render(<Main />, mountNode);
}
