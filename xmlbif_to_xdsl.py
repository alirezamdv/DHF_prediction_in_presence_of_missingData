from xml.etree import cElementTree as et
import sys
import random


COLORS = ['e6ffe6','0048BA','C0E8D5','C46210','3B7A57','FFBF00','3DDC84','915C83']

def sortchildrenby(parent, attr):
    parent[:] = sorted(parent, key=lambda child: (child.get(attr),child.get(attr)))

class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)
                    
                    
                    
class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        c = 0
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    element.tag = element.tag+'_'+str(c)
                    aDict = XmlDictConfig(element)
                    
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself 
                    element.tag = element.tag+'_'+str(c)
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    element.tag = element.tag+'_'+str(c)
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a 
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                element.tag = element.tag+'_'+str(c)
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                element.tag = element.tag+'_'+str(c)
                self.update({element.tag: element.text})
            c+=1
                  
                  
def bif_to_dsl(xmlbif_path:str)->None:
    #creating the document root
    smile = et.Element('smile')
    smile.set('version','1.0')
    smile.set('id','Network2')
    smile.set('numsamples','10000')
    smile.set('discsamples','10000')
    nodes = et.SubElement(smile, 'nodes')
    extensions = et.SubElement(smile, 'extensions')
    geni = et.SubElement(extensions,'genie')
    geni.set('version','1.0')
    geni.set('app','GeNIe 3.0.6518.0 ACADEMIC')
    geni.set('name','DHF')
    
    #reading the xmlBIF to dictionary
    tree = et.parse(xmlbif_path)
    root = tree.getroot()
    xmldict = XmlDictConfig(root)
    network=xmldict['NETWORK_0']
    def defenitions(network,element):
        for elem in network:
            if elem.split('_')[0]=='DEFINITION':
                if network[elem]['FOR_0']==element.get('id'):
                    table = [ k for k in network[elem].keys() if 'TABLE' in k]
                    probabilities = et.SubElement(element,'probabilities')
                    probabilities.text = network[elem][table[0]] 
                    par = [parent for parent in network[elem].keys() if 'GIVEN' in parent]
                    if len(par)>0:
                        parents = et.SubElement(element,'parents')
                        parents.text = ' '.join([network[elem][p] for p in par])
        return element
        
    for elem in network:
        if elem.split('_')[0]=='VARIABLE':
            print(elem)
            
            cpt = et.SubElement(nodes,'cpt')
            extention_node = et.SubElement(extensions,'node')
            cpt.set('id',network[elem]['NAME_0'])
            extention_node.set('id',network[elem]['NAME_0'])
            name = et.SubElement(extention_node,'name')
            name.text = network[elem]['NAME_0']
            interior = et.SubElement(extention_node,'interior')
            interior.set('color',random.choice(COLORS))
            outline = et.SubElement(extention_node,'outline')
            outline.set('color','000000')
            outline.set('width','3')
            font = et.SubElement(extention_node,'font')
            font.set('color','000000')
            font.set('name','Arial')
            font.set('size','14')
            position = et.SubElement(extention_node,'position')
            position.text =f'{1524} {428} {1564} {453}'
            barchart=et.SubElement(extention_node,'barchart')
            barchart.set('active','true')
            barchart.set('width','100')
            barchart.set('height','120')
            
            
            outcome = [out for out in network[elem].keys() if 'OUTCOME'in out ]
            for state in outcome:
                val = network[elem][state]
                val = val.replace('_','-') if val.startswith('_') else val
                st = et.SubElement(cpt,'state')
                st.set('id',val)
            cpt = defenitions(network,cpt)
            
    
    et.ElementTree(smile).write('DHF_bayesianNetwork_day1.xdsl',encoding='ISO-8859-1', xml_declaration=True)
    
    

if __name__=='__main__':
    
    model_path = 'day1-model.xml'
    
    bif_to_dsl(model_path)