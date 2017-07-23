import os
import selenium.webdriver as webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import contextlib
import argparse
import SimpleHTTPServer
import SocketServer
import threading
from PIL import Image
import random

def start_server(port):
    print "serving at port ", port
    Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SocketServer.TCPServer(('', port), Handler)
    thread = threading.Thread(target = httpd.serve_forever)
    thread.daemon = True
    try:
        thread.start()
    except KeyboardInterrupt:
        thread.shutdown()
        sys.exit(0)

def clean_directory(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def crop_image(file_name, width, height, bwidth, bheight, cuts=1):
    im = Image.open('{}.png'.format(file_name)) # uses PIL library to open image in memory

    for number in xrange(cuts):
        left = random.uniform(0, bwidth - width)
        top = random.uniform(0, bheight - height)
        right = left + width
        bottom = top + height

        cropped_image = im.crop((left, top, right, bottom)) # defines crop points
        cropped_image.save('{}-{}.png'.format(file_name, number))
    os.remove('{}.png'.format(file_name))

def save_image(driver, file_name):
    driver.get_screenshot_as_file('{}.png'.format(file_name))

def generate_training(FLAGS):
    driver = webdriver.Firefox()
    driver.set_window_size(FLAGS.bwidth, FLAGS.bheight)
    driver.implicitly_wait(10)
    parameters = {
        'vertical_offset': FLAGS.vertical_offset,
        'horizontal_offset': FLAGS.horizontal_offset,
        'width': FLAGS.bwidth,
        'height': FLAGS.bheight,
        'line_height': FLAGS.line_height,
        'words': FLAGS.words,
        'sentences': FLAGS.sentences,
        'skip_line': FLAGS.skip_line,
        'skip_lines': FLAGS.skip_lines,
    }
    stringified_parameters = '&'.join(['{}={}'.format(key, value) for (key, value) in parameters.items()])
    driver.get('http://127.0.0.1:8000/image_generator/?{}'.format(stringified_parameters))
    number = 0
    for number in xrange(FLAGS.images):
        wait = WebDriverWait(driver, 10)
        element = wait.until(EC.element_to_be_clickable((By.ID, 'main')))
        generate_image(driver, number, FLAGS.width, FLAGS.height, FLAGS.bwidth, FLAGS.bheight, FLAGS.cuts)
        element.click()

    if FLAGS.close:
        driver.quit()

def generate_image(driver, number, width, height, bwidth, bheight, cuts):
        file_name = '{}/{}'.format(FLAGS.training, number)
        save_image(driver, file_name)
        crop_image(file_name, width, height, bwidth, bheight, cuts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=64, help='width of the generated image')
    parser.add_argument('--height', type=int, default=64, help='height of the generated image')
    parser.add_argument('--bwidth', type=int, default=512, help='width of the browser')
    parser.add_argument('--bheight', type=int, default=512, help='height of the browset')
    parser.add_argument('--port', type=int, default=8000, help='the http port')
    parser.add_argument('--training', type=str, default='training', help='the path of the training directory')
    parser.add_argument('--images', type=int, default=10, help='the number of images in the training set')
    parser.add_argument('--vertical_offset', type=int, default=-8, help='the offset between lines')
    parser.add_argument('--horizontal_offset', type=int, default=-8, help='the offset after every sentence')
    parser.add_argument('--line_height', type=int, default=16, help='the font size of one line (same as line height)')
    parser.add_argument('--words', type=int, default=2, help='number of words in a sentence')
    parser.add_argument('--sentences', type=int, default=10, help='number of sentences in a line')
    parser.add_argument('--close', type=bool, default=True, help='close the browser at the end of the procedure')
    parser.add_argument('--web_server', type=bool, default=False, help='start the web server automatically')
    parser.add_argument('--clean', type=bool, default=True, help='delete the old training directory')
    parser.add_argument('--skip_line', type=int, default=2, help='how many lines before skipping one line')
    parser.add_argument('--skip_lines', type=int, default=2, help='how many lines to skip')
    parser.add_argument('--cuts', type=int, default=2, help='how many images from the same screenshot')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.web_server:
        start_server(FLAGS.port)
    if FLAGS.clean:
        clean_directory(FLAGS.training)
    generate_training(FLAGS)
