import os, sys
import glob
from PIL import Image
import argparse


def txtLabel_to_xmlLabel(classes_file,source_txt_path,source_img_path,save_xml_path):
    if not os.path.exists(save_xml_path):
        os.makedirs(save_xml_path)
    classes = open(classes_file).read().splitlines()
    print(classes)
    for file in os.listdir(source_txt_path):
        # img_path = os.path.join(source_img_path,file.replace('.txt','.png'))
        img_path = os.path.join(source_img_path, file.replace('.txt', '.jpg'))  # 将jpg换成png
        img_file = Image.open(img_path)
        txt_file = open(os.path.join(source_txt_path, file)).read().splitlines()
        print(txt_file)
        xml_file = open(os.path.join(save_xml_path, file.replace('.txt', '.xml')), 'w')
        width, height = img_file.size
        xml_file.write('<annotation>\n')
        xml_file.write('\t<folder>simple</folder>\n')
        xml_file.write('\t<filename>' + str(file) + '</filename>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(width) + ' </width>\n')
        xml_file.write('\t\t<height>' + str(height) + '</height>\n')
        xml_file.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml_file.write('\t</size>\n')

        for line in txt_file:
            print(line)
            line_split = line.split(' ')
            x_center = float(line_split[1])
            y_center = float(line_split[2])
            w = float(line_split[3])
            h = float(line_split[4])
            xmax = int((2*x_center*width + w*width)/2)
            xmin = int((2*x_center*width - w*width)/2)
            ymax = int((2*y_center*height + h*height)/2)
            ymin = int((2*y_center*height - h*height)/2)

            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>'+ str(classes[int(line_split[0])]) +'</name>\n')
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
        xml_file.write('</annotation>')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes_file', type=str, default="person.names")
    parser.add_argument('--source_txt_path', type=str, default="")
    parser.add_argument('--source_img_path', type=str, default="")
    parser.add_argument('--save_xml_path', type=str, default="")
    opt = parser.parse_args()

    txtLabel_to_xmlLabel(opt.classes_file, opt.source_txt_path, opt.source_img_path, opt.save_xml_path)
