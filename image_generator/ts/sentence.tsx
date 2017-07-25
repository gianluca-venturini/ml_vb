import * as React from "react";
import * as ReactDOM from "react-dom";

interface Props {
  color: string,
  horizontalOffset: number,
  verticalOffset: number,
  lineHeight: number,
  opacity: number,
  fontSize: number,
  fontWeight: 200 | 400 | 600 | 800,
}

export class Sentence extends React.Component<Props, {}> {
  render() {
    return (
      <span
        id='text'
        style={{
          color: this.props.color,
          position: 'relative',
          whiteSpace: 'nowrap',
          marginRight: this.props.horizontalOffset,
          marginTop: this.props.verticalOffset,
          display: 'inline-block',
          height: this.props.lineHeight,
          lineHeight: `${this.props.lineHeight}px`,
          opacity: this.props.opacity,
          fontSize: this.props.fontSize,
          fontWeight: this.props.fontWeight,
        }}
      >{this.props.children}</span>
    )
  }
}

