/**
 * Created by Emmitt on 3/3/2017.
 */
/**
 * Created by Emmitt on 3/3/2017.
 */
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class SuffixTree{

    //Constructor which create a suffix tree from source string
    public SuffixTree(String source){
        this.root = createSuffixTree(source,new Node(), new ArrayList<Node>()).get(0);
    }

    //Each node in the tree has a char type currentValue and a map which holds all children of this node
    class Node{
        private char currentValue;
        private Map<Character,Node> children;
        //Several constructor
        Node(){
            this.currentValue = '*';
            this.children = new HashMap<Character,Node>();
        }
        Node(char currentValue){
            this.currentValue = currentValue;
            this.children = new HashMap<Character,Node>();
        }
        //Get the value of root
        char getValue(){
            return this.currentValue;
        }

        //Add a child to node c
        void addChild(Node c) {
            this.children.put(c.getValue(),c);
        }
        //Return true if current node has a child c
        boolean hasChild(Node c) {
            return this.children.containsKey(c.getValue());
        }
        //Return all childern of node c
        Node getChild(Node c){
            return this.children.get(c.getValue());
        }

    }

    public Node root;
    /*
     * Helper method that initializes the suffix tree
     */
    //Add a new String "source" to the tree
    public List<Node> createSuffixTree(String source, Node root, List<Node>list){
        boolean flag=true;
        List<Node> links=new ArrayList<Node>();
        links.add(root);
        for(int i=0;i<source.length();i++){
            Node parent = new Node(source.charAt(i));
            if(root.hasChild(parent)){
                parent = root.getChild(parent);
            }
             Node current = parent;
            if(i==source.length()-1)
                links.add(current);

            for(int j=i+1;j<source.length();j++){
                Node temp = new Node(source.charAt(j));
                if(current.hasChild(temp)){//if true, this is two identical char in row, like aa, bb, cc
                    temp = current.getChild(temp);
                }else{
                    current.addChild(temp);
                }
                current = temp;
                if(j==source.length()-1)
                    links.add(current);
            }
            root.addChild(parent);
            for(int k=1;k<list.size()&&flag;k++) {
                list.get(k).children.put(parent.currentValue, parent);
            }
            flag=false;
        }
        return links;
    }


    //Search function, check if target string exists in the tree
    boolean search(String target){
        Map<Character,Node> rootChildren = this.root.children;
        for(char c:target.toCharArray()){
            if(rootChildren.containsKey(c)){
                rootChildren = rootChildren.get(c).children;
            }else{
                return false;
            }
        }
        return true;
    }
}

