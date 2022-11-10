import xml.etree.ElementTree as ET


class UrdfModifier(object):
    def __init__(self, folder, file_name):
        """

        :param folder_path: Folder where the urdf is located, must end with "/"
        :param file_name: File name without extension and folder (example: "chair")
        """
        self.folder = folder
        self.file_name = file_name
        self.file_path = self.folder + self.file_name + ".urdf"
        self.parsed_xml = ET.parse(self.file_path)
        self.xml_root = self.parsed_xml.getroot()

    def save_urdf(self, file_name):
        """
        :param path: Path where to save new urdf (omit the .urdf)
        :return: Saved file path
        """
        save_path = self.folder + file_name + ".urdf"
        self.parsed_xml.write(save_path)
        return save_path

    def set_scale(self, x, y, z):
        """
        Set scale of the object's urdf.
        :param x: x scale
        :param y: y scale
        :param z: z scale
        :return: None
        """
        for mesh in self.xml_root.iter('mesh'):
            mesh.set('scale', str(x) + " " + str(y) + " " + str(z))

        for mesh in self.xml_root.iter('box'):
            mesh.set('size', str(x) + " " + str(y) + " " + str(z))

    def get_file_path(self):
        """
        :return: Original file path.
        """
        return self.file_path
