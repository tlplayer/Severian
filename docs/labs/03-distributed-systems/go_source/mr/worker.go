package mr

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net/rpc"
	"os"
	"sort"
	"time"
)

// Map functions return a slice of KeyValue.
type KeyValue struct {
	Key   string
	Value string
}

// for sorting by key.
type ByKey []KeyValue

// for sorting by key.
func (a ByKey) Len() int           { return len(a) }
func (a ByKey) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByKey) Less(i, j int) bool { return a[i].Key < a[j].Key }

// use ihash(key) % NReduce to choose the reduce
// task number for each KeyValue emitted by Map.
func ihash(key string) int {
	h := fnv.New32a()
	h.Write([]byte(key))
	return int(h.Sum32() & 0x7fffffff)
}

// main/mrworker.go calls this function.
func Worker(mapf func(string, string) []KeyValue,
	reducef func(string, []string) string) {

	for {
		args := RequestTaskArgs{}
		reply := RequestTaskReply{}

		if !call("Coordinator.RequestTask", &args, &reply) {
			time.Sleep(time.Second)
			continue
		}

		switch reply.Type {
		case MapTask:
			executeMap(reply, mapf)
			complete(reply.TaskID, MapTask)

		case ReduceTask:
			executeReduce(reply, reducef)
			complete(reply.TaskID, ReduceTask)

		case WaitTask:
			time.Sleep(time.Second)

		case ExitTask:
			return
		}
	}
}

func executeMap(task RequestTaskReply, mapf func(string, string) []KeyValue) {
	content, err := os.ReadFile(task.File)
	if err != nil {
		log.Fatalf("cannot read %v: %v", task.File, err)
	}

	kva := mapf(task.File, string(content))

	files := make([]*os.File, task.NReduce)
	encoders := make([]*json.Encoder, task.NReduce)
	tmpNames := make([]string, task.NReduce)

	for i := 0; i < task.NReduce; i++ {
		file, err := os.CreateTemp(".", "mr-map-tmp-")
		if err != nil {
			log.Fatalf("cannot create map temp file: %v", err)
		}

		files[i] = file
		encoders[i] = json.NewEncoder(file)
		tmpNames[i] = file.Name()
	}

	for _, kv := range kva {
		reduceID := ihash(kv.Key) % task.NReduce
		if err := encoders[reduceID].Encode(&kv); err != nil {
			log.Fatalf("cannot encode intermediate key/value: %v", err)
		}
	}

	for i, file := range files {
		if err := file.Close(); err != nil {
			log.Fatalf("cannot close map temp file: %v", err)
		}

		finalName := fmt.Sprintf("mr-%d-%d", task.TaskID, i)
		if err := os.Rename(tmpNames[i], finalName); err != nil {
			log.Fatalf("cannot rename %v to %v: %v", tmpNames[i], finalName, err)
		}
	}
}

func executeReduce(task RequestTaskReply, reducef func(string, []string) string) {
	intermediate := []KeyValue{}

	for mapID := 0; mapID < task.NMap; mapID++ {
		fileName := fmt.Sprintf("mr-%d-%d", mapID, task.TaskID)
		file, err := os.Open(fileName)
		if err != nil {
			log.Fatalf("cannot open %v: %v", fileName, err)
		}

		decoder := json.NewDecoder(file)
		for {
			var kv KeyValue
			err := decoder.Decode(&kv)
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatalf("cannot decode %v: %v", fileName, err)
			}

			intermediate = append(intermediate, kv)
		}

		if err := file.Close(); err != nil {
			log.Fatalf("cannot close %v: %v", fileName, err)
		}
	}

	sort.Sort(ByKey(intermediate))

	output, err := os.CreateTemp(".", "mr-out-tmp-")
	if err != nil {
		log.Fatalf("cannot create reduce temp file: %v", err)
	}

	i := 0
	for i < len(intermediate) {
		j := i + 1
		for j < len(intermediate) && intermediate[j].Key == intermediate[i].Key {
			j++
		}

		values := []string{}
		for k := i; k < j; k++ {
			values = append(values, intermediate[k].Value)
		}

		result := reducef(intermediate[i].Key, values)
		if _, err := fmt.Fprintf(output, "%v %v\n", intermediate[i].Key, result); err != nil {
			log.Fatalf("cannot write reduce output: %v", err)
		}

		i = j
	}

	if err := output.Close(); err != nil {
		log.Fatalf("cannot close reduce temp file: %v", err)
	}

	finalName := fmt.Sprintf("mr-out-%d", task.TaskID)
	if err := os.Rename(output.Name(), finalName); err != nil {
		log.Fatalf("cannot rename %v to %v: %v", output.Name(), finalName, err)
	}
}

func complete(taskID int, taskType TaskType) {
	args := CompleteTaskArgs{
		Type:   taskType,
		TaskID: taskID,
	}
	reply := CompleteTaskReply{}

	call("Coordinator.CompleteTask", &args, &reply)
}

// send an RPC request to the coordinator, wait for the response.
// usually returns true.
// returns false if something goes wrong.
func call(rpcname string, args interface{}, reply interface{}) bool {
	// c, err := rpc.DialHTTP("tcp", "127.0.0.1"+":1234")
	sockname := coordinatorSock()
	c, err := rpc.DialHTTP("unix", sockname)
	if err != nil {
		log.Fatal("dialing:", err)
	}
	defer c.Close()

	err = c.Call(rpcname, args, reply)
	if err == nil {
		return true
	}

	fmt.Println(err)
	return false
}
