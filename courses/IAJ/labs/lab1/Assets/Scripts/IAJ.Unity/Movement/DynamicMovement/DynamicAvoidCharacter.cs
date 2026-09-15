using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public class DynamicAvoidCharacter : DynamicMovement
    {
        public override string Name
        {
            get { return "Avoid Character"; }
        }

        public float MaxTimeLookAhead { get; set; }
        public float AvoidMargin { get; set; }


        public DynamicAvoidCharacter(KinematicData target)
        {
            this.Target = target;
            this.MaxTimeLookAhead = 2.0f;
            this.AvoidMargin = 3.0f;

            this.Output = new MovementOutput();
        }

        public override MovementOutput GetMovement()
        {
            this.Output.Clear();

            var deltaPos = this.Target.Position - this.Character.Position;
            var deltaVel = this.Target.velocity - this.Character.Velocity;
            var deltaSqrSpeed = deltaVel.sqrMagnitude;
            var collisionDistance = 2 * this.AvoidMargin;
            var currentDistance = deltaPos.magnitude;

            //TODO
          

            this.Output.linear.Normalize();
            this.Output.linear *= this.Character.MaxAcceleration;
            
            return this.Output;
            
        }
    }
}
