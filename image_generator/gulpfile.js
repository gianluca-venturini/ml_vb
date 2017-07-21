var gulp = require('gulp');
var source = require('vinyl-source-stream');
var browserify = require('browserify');
var tsify = require('tsify');
var debowerify = require('debowerify');

var config = {
	publicPath: __dirname + '/js',
	app: {
		path: __dirname + '/ts',
		main: 'index.ts',
		result: 'application.js'
	}
};

gulp.task('default', function() {
	var bundler = browserify({basedir: config.app.path})
		.add(config.app.path + '/' + config.app.main)
		.plugin(tsify, {
      noImplicitAny: true,
      jsx: 'react',
      lib: ["es2015", "es2017", "dom"]
    })
    .transform(debowerify);

	return bundler.bundle()
		.pipe(source(config.app.result))
		.pipe(gulp.dest(config.publicPath));
});
