import json
from os import listdir
import os
from os.path import join, isdir, isfile
import sys


def get_directories(directory):
    return [d for d in listdir(directory) if isdir(join(directory, d)) and d != "__pycache__"]


def get_files(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and f != "__init__.py"]


def get_classes(concept_file):
    package = ".".join(concept_file.split("/")[-7:-1]) + "."
    file = os.path.basename(concept_file)

    classes = []
    # Check that the file is a python file but not the init.py
    if not file.endswith('.py') or file == '__init__.py':
        return classes

    # Get the class and module
    class_name = file[:-3]
    class_module = __import__(package + class_name, fromlist=[class_name])

    # Get the frames' class
    module_dict = class_module.__dict__
    for obj in module_dict:
        if isinstance(module_dict[obj], type) and module_dict[obj].__module__ == class_module.__name__:
            classes.append(obj)
    return classes


def create_ontology_json(ontology_directory):
    """
    Creating the ontology json from the file system
    """
    ontology = {}
    versions = get_directories(ontology_directory)
    for version in versions:
        version_directory = ontology_directory + "/" + version
        ontology[version] = {}
        variable_types = get_directories(version_directory)
        for variable_type in variable_types:
            variable_type_directory = version_directory + "/" + variable_type
            ontology[version][variable_type] = {}
            namespaces = get_directories(variable_type_directory)
            for namespace in namespaces:
                namespace_directory = variable_type_directory + "/" + namespace
                ontology[version][variable_type][namespace] = {}
                domains = get_directories(namespace_directory)
                for domain in domains:
                    domain_directory = namespace_directory + "/" + domain
                    ontology[version][variable_type][namespace][domain] = {}
                    concepts = get_files(domain_directory)
                    for concept in concepts:
                        concept_file = domain_directory + "/" + concept
                        variables = get_classes(concept_file)
                        concept = concept.replace(".py", "")
                        ontology[version][variable_type][namespace][domain][concept] = variables
    with open(ontology_directory + "/ontology.json", "w+") as ontology_file:
        json.dump(ontology, ontology_file, indent=2)


if __name__ == '__main__':
    # Entry point creating the ontology json from the file system
    create_ontology_json(sys.argv[1])
