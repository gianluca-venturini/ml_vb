import * as React from "react";
import * as ReactDOM from "react-dom";

interface Props {
  color: string,
  horizontalOffset: number,
  verticalOffset: number,
  lineHeight: number,
  opacity: number,
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
          fontSize: this.props.lineHeight,
          display: 'inline-block',
          height: this.props.lineHeight,
          opacity: this.props.opacity,
        }}
      >{this.props.children}</span>
    )
  }
}

