"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
exports.__esModule = true;
var React = require("react");
var ReactDOM = require("react-dom");
var Word = (function (_super) {
    __extends(Word, _super);
    function Word() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Word.prototype.render = function () {
        return (React.createElement("span", null, "Text"));
    };
    return Word;
}(React.Component));
var mountNode = document.getElementById('root');
ReactDOM.render(React.createElement(Word, null), mountNode);
